import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: np.ndarray             # 5x2 float32 array: [right_eye, left_eye, nose, right_mouth, left_mouth]
    score: float                      # Confidence score [0, 1]


class YuNetDetector:
    """
    YuNet Face Detector using OpenCV's built-in FaceDetectorYN.
    Detects faces and predicts 5 keypoints (eyes, nose, mouth corners).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.40,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at {model_path}")

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=0,
            target_id=0
        )
        self.current_input_size = (320, 320)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a BGR image.
        Returns a list of DetectedFace objects.
        """
        h, w = image.shape[:2]
        if w < 10 or h < 10:
            return []

        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        # Contrast normalization on luminance (fixes underexposed faces and backlit webcam shadows)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        _, raw_faces = self.detector.detect(enhanced)
        if raw_faces is None or len(raw_faces) == 0:
            _, raw_faces = self.detector.detect(image)

        if raw_faces is None or len(raw_faces) == 0:
            return []

        results = []
        for face in raw_faces:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, fw, fh = map(int, face[:4])
            landmarks = face[4:14].reshape((5, 2)).astype(np.float32)
            score = float(face[14])

            # Bound coords to image limits
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)

            results.append(DetectedFace(
                bbox=(x, y, fw, fh),
                landmarks=landmarks,
                score=score
            ))

        return results

    @staticmethod
    def draw(image: np.ndarray, faces: List[DetectedFace], color=(0, 255, 0)) -> np.ndarray:
        """
        Draw bounding boxes, landmarks, and confidence score on image.
        """
        vis = image.copy()
        colors_lm = [
            (255, 0, 0),    # right eye (blue)
            (0, 0, 255),    # left eye (red)
            (0, 255, 0),    # nose (green)
            (255, 255, 0),  # right mouth corner (cyan)
            (0, 255, 255),  # left mouth corner (yellow)
        ]

        for face in faces:
            x, y, w, h = face.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            label = f"{face.score:.2f}"
            cv2.putText(vis, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for i, (lx, ly) in enumerate(face.landmarks):
                cv2.circle(vis, (int(lx), int(ly)), 3, colors_lm[i % len(colors_lm)], -1)

        return vis
