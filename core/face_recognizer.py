import os
from typing import Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

# Standard ArcFace 112x112 target landmark coordinates
# Order: right eye, left eye, nose tip, right mouth corner, left mouth corner
ARCFACE_REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class AuraFaceRecognizer:
    """
    AuraFace Feature Extractor and Matcher.
    Uses glintr100.onnx to extract 512-dimensional face embeddings.
    """
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "glintr100.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AuraFace model not found at {model_path}")

        self.model_path = model_path
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align and crop face to 112x112 using 5 landmarks via similarity transform.
        image: BGR image
        landmarks: 5x2 array of keypoints
        returns: 112x112 aligned BGR face crop
        """
        # Estimate similarity transformation matrix (2x3)
        M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_REFERENCE_LANDMARKS_112)
        if M is None:
            # Fallback if alignment fails
            return cv2.resize(image, (112, 112))

        aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
        return aligned

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim normalized embedding from an aligned face.
        image: BGR image
        landmarks: 5x2 array of keypoints
        returns: normalized numpy vector of shape (512,)
        """
        aligned_bgr = self.align_face(image, landmarks)
        # Convert BGR to RGB
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]: (pixel - 127.5) / 127.5
        input_tensor = (aligned_rgb.astype(np.float32) - 127.5) / 127.5
        # Reshape to (1, 3, 112, 112) [NCHW]
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_embedding = outputs[0][0]

        # L2 normalize
        norm = np.linalg.norm(raw_embedding)
        if norm == 0:
            return raw_embedding
        normalized_embedding = raw_embedding / norm
        return normalized_embedding

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized embeddings.
        Returns value in [-1.0, 1.0]. A score >= 0.45-0.50 typically indicates the same person.
        """
        return float(np.dot(emb1, emb2))
