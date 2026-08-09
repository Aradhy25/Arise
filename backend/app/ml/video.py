"""Video frame extraction and sampling with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoSample:
    frames: list[np.ndarray]
    frame_indices: list[int]
    total_frames: int
    fps: float
    duration_sec: float


def sample_video_frames(path: str, max_frames: int = 40) -> VideoSample:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    duration = total / fps if fps > 0 and total > 0 else 0.0

    if total <= 0:
        # Fallback: read sequentially
        frames_bgr: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
        cap.release()
        total = len(frames_bgr)
        if total == 0:
            raise ValueError("Video contains no readable frames")
        indices = _uniform_indices(total, max_frames)
        rgb = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in indices]
        return VideoSample(rgb, indices, total, fps, duration)

    indices = _uniform_indices(total, max_frames)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError("Failed to extract frames from video")

    return VideoSample(frames, indices[: len(frames)], total, fps, duration)


def _uniform_indices(total: int, max_frames: int) -> list[int]:
    n = min(max_frames, total)
    if n <= 1:
        return [0]
    return [int(i * (total - 1) / (n - 1)) for i in range(n)]
