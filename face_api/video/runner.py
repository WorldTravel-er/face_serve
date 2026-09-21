from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from face_api.video.analysis import VideoAnalysisProcessor


class VideoAnalysisRunner:
    def __init__(
        self,
        processor: VideoAnalysisProcessor,
        max_concurrency: int = 1,
    ) -> None:
        max_concurrency = int(max_concurrency)
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.processor = processor
        self.max_concurrency = max_concurrency
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="video-analysis",
        )

    def submit_task(self, task_name: str) -> Future[None]:
        return self._executor.submit(self.run_task, task_name)

    def run_task(self, task_name: str) -> None:
        self.processor.run(task_name)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
