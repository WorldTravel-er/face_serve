from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_channels: int, out_channels: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def _conv_bn_1x1(in_channels: int, out_channels: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_dw(in_channels: int, out_channels: int, stride: int, leaky: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    """MobileNet0.25 backbone used by the supplied RetinaFace checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            _conv_bn(3, 8, 2, leaky=0.1),
            _conv_dw(8, 16, 1),
            _conv_dw(16, 32, 2),
            _conv_dw(32, 32, 1),
            _conv_dw(32, 64, 2),
            _conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            _conv_dw(64, 128, 2),
            _conv_dw(128, 128, 1),
            _conv_dw(128, 128, 1),
            _conv_dw(128, 128, 1),
            _conv_dw(128, 128, 1),
            _conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            _conv_dw(128, 256, 2),
            _conv_dw(256, 256, 1),
        )

    def forward(self, value: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        stage1 = self.stage1(value)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return OrderedDict((("1", stage1), ("2", stage2), ("3", stage3)))


class FPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = _conv_bn_1x1(in_channels[0], out_channels, leaky=leaky)
        self.output2 = _conv_bn_1x1(in_channels[1], out_channels, leaky=leaky)
        self.output3 = _conv_bn_1x1(in_channels[2], out_channels, leaky=leaky)
        self.merge1 = _conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = _conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, values: OrderedDict[str, torch.Tensor]) -> list[torch.Tensor]:
        value1, value2, value3 = list(values.values())
        output1 = self.output1(value1)
        output2 = self.output2(value2)
        output3 = self.output3(value3)
        output2 = self.merge2(output2 + F.interpolate(output3, size=output2.shape[2:], mode="nearest"))
        output1 = self.merge1(output1 + F.interpolate(output2, size=output1.shape[2:], mode="nearest"))
        return [output1, output2, output3]


class SSH(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4:
            raise ValueError("SSH out_channels must be divisible by four")
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.conv3X3 = _conv_bn_no_relu(in_channels, out_channels // 2)
        self.conv5X5_1 = _conv_bn(in_channels, out_channels // 4, leaky=leaky)
        self.conv5X5_2 = _conv_bn_no_relu(out_channels // 4, out_channels // 4)
        self.conv7X7_2 = _conv_bn(out_channels // 4, out_channels // 4, leaky=leaky)
        self.conv7x7_3 = _conv_bn_no_relu(out_channels // 4, out_channels // 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value3 = self.conv3X3(value)
        value5_1 = self.conv5X5_1(value)
        value5 = self.conv5X5_2(value5_1)
        value7 = self.conv7x7_3(self.conv7X7_2(value5_1))
        return F.relu(torch.cat((value3, value5, value7), dim=1))


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 64, anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, anchors * 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv1x1(value).permute(0, 2, 3, 1).contiguous()
        return output.view(output.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 64, anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, anchors * 4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv1x1(value).permute(0, 2, 3, 1).contiguous()
        return output.view(output.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 64, anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, anchors * 10, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv1x1(value).permute(0, 2, 3, 1).contiguous()
        return output.view(output.shape[0], -1, 10)


class RetinaFace(nn.Module):
    """Inference-only RetinaFace network with a supported backbone."""

    def __init__(self, backbone: str = "mobilenet0.25") -> None:
        super().__init__()
        if backbone == "mobilenet0.25":
            self.body = MobileNetV1()
            fpn_channels = (64, 128, 256)
            output_channels = 64
        elif backbone == "resnet50":
            from torchvision.models import resnet50
            from torchvision.models._utils import IntermediateLayerGetter

            self.body = IntermediateLayerGetter(
                resnet50(weights=None),
                return_layers={"layer2": "1", "layer3": "2", "layer4": "3"},
            )
            fpn_channels = (512, 1024, 2048)
            output_channels = 256
        else:
            raise ValueError(f"Unsupported RetinaFace backbone: {backbone!r}")
        self.fpn = FPN(fpn_channels, output_channels)
        self.ssh1 = SSH(output_channels, output_channels)
        self.ssh2 = SSH(output_channels, output_channels)
        self.ssh3 = SSH(output_channels, output_channels)
        self.ClassHead = nn.ModuleList(ClassHead(output_channels) for _ in range(3))
        self.BboxHead = nn.ModuleList(BboxHead(output_channels) for _ in range(3))
        self.LandmarkHead = nn.ModuleList(LandmarkHead(output_channels) for _ in range(3))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pyramid = self.fpn(self.body(inputs))
        features = (self.ssh1(pyramid[0]), self.ssh2(pyramid[1]), self.ssh3(pyramid[2]))
        boxes = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        scores = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        landmarks = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        return boxes, F.softmax(scores, dim=-1), landmarks


__all__ = ["RetinaFace"]
