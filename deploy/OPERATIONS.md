# Face Serve systemd 与离线运维手册

本目录是主机部署的唯一事实来源，内容刻意独立于仓库中已有的旧部署说明。

## 支持的发布配置

| 配置 | 主机环境 | 推理方式 |
| --- | --- | --- |
| rk3588-rknn | Linux aarch64/RK3588、Python 3.11 或 3.12、RKNN Lite2 2.3.2 | RKNN 检测与识别、ONNX CPU 对齐 |
| linux-x86_64-onnx-cpu | Linux x86_64、Python 3.11 或 3.12 | ONNX CPU |
| linux-x86_64-onnx-cuda | Linux x86_64，以及经过验证的 NVIDIA 驱动栈 | ONNX CUDA |

CUDA requirements 文件必须明确安装能够提供 `CUDAExecutionProvider` 的 ONNX Runtime 版本。不得将仅支持 CPU 的依赖锁文件重新标记为 CUDA 配置。

## 运行目录布局

- 程序版本：`/opt/face-serve/releases/<version>`
- 当前生效版本：`/opt/face-serve/current`
- 配置文件：`/etc/face-serve`
- 持久化数据：`/var/lib/face-serve`
- 日志：journald，以及可选的 `/var/log/face-serve`
- 备份：`/var/backups/face-serve`
- 辅助程序：`/usr/libexec/face-serve`

## 准备依赖

请在干净的构建主机或容器中执行以下命令。构建环境必须与目标环境具有相同的 CPU 架构、操作系统系列、Python ABI 和 glibc 版本级别：

```bash
deploy/scripts/prepare-offline-dependencies.sh \
  --profile linux-x86_64-onnx-cpu \
  --output-dir build/dependencies \
  --with-debs
```

对于 RKNN，请在兼容的 aarch64 主机上执行，并确保能够获取 `rknn-toolkit-lite2 2.3.2` wheel。对于 CUDA，请传入经过审核的 requirements 文件，其中应包含 `onnxruntime-gpu`，且版本必须与目标主机的驱动兼容。

## 构建离线包

```bash
deploy/scripts/build-offline-package.sh \
  --profile linux-x86_64-onnx-cpu \
  --version 1.0.0 \
  --wheelhouse build/dependencies/wheelhouse \
  --deb-dir build/dependencies/debs
```

该命令会在 `dist/` 中生成一个 `.tar.zst` 压缩包及其 SHA-256 校验文件。包内的每个文件都由内部 `SHA256SUMS` 覆盖校验。

## 安装与升级

将压缩包复制到断网的目标主机，然后执行：

```bash
tar --zstd -xf face-serve-1.0.0-linux-x86_64-onnx-cpu.tar.zst
sudo ./face-serve-1.0.0-linux-x86_64-onnx-cpu/deploy/scripts/install.sh \
  ./face-serve-1.0.0-linux-x86_64-onnx-cpu
```

安装器会在修改主机之前校验文件摘要和系统架构。它会先完成新版本程序和虚拟环境的暂存，再停止当前运行的服务。

升级过程中，安装器会备份数据、原子切换 `current` 链接、执行预检并启动服务。如果预检或就绪检查失败，安装器会恢复到之前的版本链接。

安装器会保留已有的 `/etc/face-serve/face-serve.env`。每次升级后都应将现有配置与新版本提供的示例配置进行比较。

## 服务运维

```bash
sudo facectl start
sudo facectl stop
sudo facectl restart
facectl status
facectl health
facectl doctor
sudo facectl backup --label before-maintenance
facectl versions
sudo facectl rollback 1.0.0
journalctl -u face-serve.service
```

本地依赖服务必须显式配置在 `/etc/face-serve/dependencies.conf` 白名单中。使用 `facectl deps status|start|stop|restart` 管理这些服务。

远程摄像头不是 systemd 依赖。摄像头离线只会使实时识别进入降级状态，不代表整个人脸 API 服务故障。

## 健康检查行为

- `/healthz` 表示进程存活状态。
- `/readyz` 表示模型和数据库是否就绪；检查失败时返回 HTTP 503。
- 定时器每 30 秒执行一次就绪检查。
- 处于运行状态但不健康的 API 连续失败三次后会被重启。
- 管理员主动停止的服务不会被健康检查定时器重新拉起。
- systemd 的重启频率限制可防止服务陷入永久重启循环。
- RTSP 断流不会导致就绪检查失败；采集工作线程会独立执行重连。

## RKNN 检查

预检会检查系统架构、模型及对齐模型哈希、`librknnrt`、render 设备、访问权限、`rknnlite` 导入以及准确的 Lite2 版本。正式发布前应执行更深入的设备推理检查：

```bash
sudo facectl preflight --deep-rknn-check
```

服务用户必须属于 `render` 和 `video` 用户组。如果开发板提供的 render 节点不是 `/dev/dri/renderD129`，需要修改对应的设备 drop-in 配置。

## 备份与回滚

备份内容包括 SQLite 在线备份、人员图片目录和视频数据目录。程序版本回滚不会自动恢复业务数据。

如果某个版本包含破坏性数据库结构迁移，必须明确声明，并在打包前提供经过验证的数据恢复或迁移路径。

更换识别模型可能导致已保存的人脸特征向量失效。此类版本必须先重建人脸库特征或迁移数据，确认完成后才能接入业务流量。

## 卸载

默认卸载会保留程序版本、配置、数据和备份：

```bash
sudo /usr/libexec/face-serve/uninstall.sh
```

如需执行破坏性清理，必须显式指定参数：

```bash
sudo /usr/libexec/face-serve/uninstall.sh --purge-releases --purge-data
```

## 发布前验收

创建 GitHub Release 前必须完成以下工作：

1. 运行完整的 Python 测试套件。
2. 对每个 shell 脚本执行 `bash -n`。
3. 对 systemd unit 文件执行 `systemd-analyze verify`。
4. 在干净且完全断网的目标主机上完成安装。
5. 测试模型缺失、校验值损坏、NPU 设备缺失、端口冲突、进程被终止、RTSP 断流与恢复、升级失败以及版本回滚。
6. 完成一次完整的图片人脸匹配和一次实时 WebSocket 人脸识别。
7. 保存测试报告、安装包哈希、SBOM 和许可证清单。
