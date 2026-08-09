"""Face detection — OpenCV DNN (portable) with optional MediaPipe."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.ml.preprocessing import FaceCrop


class FaceDetector:
    """Detect faces and return cropped RGB face regions.

    Uses OpenCV's YuNet / Haar cascade as a reliable default that does not
    require GPU or heavyweight RetinaFace wheels. When insightface/RetinaFace
    is installed, prefer that for higher quality.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detector = None
        self._backend = "haar"
        self._init_detector()

    def _init_detector(self) -> None:
        # Prefer OpenCV YuNet if model file is present
        yunet = Path(__file__).resolve().parents[2] / "weights" / "face_detection_yunet_2023mar.onnx"
        if yunet.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(yunet), "", (320, 320), self.min_confidence, 0.3, 5000
                )
                self._backend = "yunet"
                return
            except Exception:
                pass

        # Try MediaPipe if available
        try:
            import mediapipe as mp  # type: ignore

            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.min_confidence
            )
            self._backend = "mediapipe"
            return
        except Exception:
            pass

        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade)
        self._backend = "haar"

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, image_rgb: np.ndarray, max_faces: int = 5) -> list[FaceCrop]:
        h, w = image_rgb.shape[:2]
        faces: list[FaceCrop] = []

        if self._backend == "yunet":
            self._detector.setInputSize((w, h))
            _, results = self._detector.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            if results is not None:
                for det in results[:max_faces]:
                    x, y, bw, bh = map(int, det[:4])
                    conf = float(det[-1])
                    faces.append(self._crop(image_rgb, x, y, bw, bh, conf))
            return faces

        if self._backend == "mediapipe":
            results = self._detector.process(image_rgb)
            if results.detections:
                for det in results.detections[:max_faces]:
                    box = det.location_data.relative_bounding_box
                    x = int(box.xmin * w)
                    y = int(box.ymin * h)
                    bw = int(box.width * w)
                    bh = int(box.height * h)
                    conf = float(det.score[0]) if det.score else 0.0
                    faces.append(self._crop(image_rgb, x, y, bw, bh, conf))
            return faces

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        rects = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, bw, bh) in list(rects)[:max_faces]:
            faces.append(self._crop(image_rgb, int(x), int(y), int(bw), int(bh), 1.0))
        return faces

    def detect_or_full(self, image_rgb: np.ndarray) -> FaceCrop:
        faces = self.detect(image_rgb, max_faces=1)
        if faces:
            return faces[0]
        h, w = image_rgb.shape[:2]
        return FaceCrop(image_rgb=image_rgb, bbox=(0, 0, w, h), confidence=0.0)

    @staticmethod
    def _crop(image_rgb: np.ndarray, x: int, y: int, bw: int, bh: int, conf: float) -> FaceCrop:
        h, w = image_rgb.shape[:2]
        pad = int(0.2 * max(bw, bh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        crop = image_rgb[y1:y2, x1:x2].copy()
        return FaceCrop(image_rgb=crop, bbox=(x1, y1, x2 - x1, y2 - y1), confidence=conf)
