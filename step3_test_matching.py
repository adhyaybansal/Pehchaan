"""
Step 3: Face Matching with AuraFace (glintr100)
Compares two face photos, extracts 512-dim embeddings, and computes cosine similarity.

Usage:
  python step3_test_matching.py --img1 path/to/photo1.jpg --img2 path/to/photo2.jpg
Or run interactively to capture 2 snapshots directly from your webcam!
"""

import os
import argparse
import cv2
import numpy as np
from core.face_detector import YuNetDetector
from core.face_recognizer import AuraFaceRecognizer


def extract_single_face_embedding(detector: YuNetDetector, recognizer: AuraFaceRecognizer, image: np.ndarray):
    faces = detector.detect(image)
    if len(faces) == 0:
        return None, None
    # Pick the face with highest confidence score
    best_face = max(faces, key=lambda f: f.score)
    emb = recognizer.get_embedding(image, best_face.landmarks)
    aligned = recognizer.align_face(image, best_face.landmarks)
    return emb, aligned


def capture_snapshot(cap, title="Press 's' to capture, 'q' to cancel"):
    print(f"Instructions: {title}")
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        vis = frame.copy()
        cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("AuraFace Snapshot Capture", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None


def run_interactive_matching(camera_index: int = 0, threshold: float = 0.50):
    detector = YuNetDetector()
    recognizer = AuraFaceRecognizer()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Error] Camera {camera_index} could not be opened.")
        return

    print("\n--- CAPTURING PHOTO 1 ---")
    img1 = capture_snapshot(cap, "Photo 1: Look at camera and press 's'")
    if img1 is None:
        cap.release()
        return

    print("\n--- CAPTURING PHOTO 2 ---")
    img2 = capture_snapshot(cap, "Photo 2: Change pose slightly and press 's'")
    cap.release()
    if img2 is None:
        return

    compare_images(detector, recognizer, img1, img2, threshold=threshold)


def compare_images(
    detector: YuNetDetector,
    recognizer: AuraFaceRecognizer,
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float = 0.50
):
    emb1, aligned1 = extract_single_face_embedding(detector, recognizer, img1)
    if emb1 is None:
        print("[Error] No face detected in Photo 1!")
        return

    emb2, aligned2 = extract_single_face_embedding(detector, recognizer, img2)
    if emb2 is None:
        print("[Error] No face detected in Photo 2!")
        return

    similarity = recognizer.compute_similarity(emb1, emb2)
    is_match = similarity >= threshold

    print("\n" + "=" * 50)
    print(f"Cosine Similarity : {similarity:.4f}")
    print(f"Decision Threshold: {threshold:.2f}")
    if is_match:
        print(f"VERDICT           : [MATCH] Access Granted! (Same person)")
    else:
        print(f"VERDICT           : [NO MATCH] Access Denied! (Different person)")
    print("=" * 50 + "\n")

    # Display aligned faces side by side
    combined = np.hstack([aligned1, aligned2])
    color = (0, 255, 0) if is_match else (0, 0, 255)
    result_text = f"Similarity: {similarity:.2f} ({'MATCH' if is_match else 'NO MATCH'})"
    
    # Scale up for easy viewing
    display_view = cv2.resize(combined, (448, 224), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display_view, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Aligned Face Crops Comparison (Press any key to close)", display_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_file_matching(img1_path: str, img2_path: str, threshold: float = 0.50):
    detector = YuNetDetector()
    recognizer = AuraFaceRecognizer()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"[Error] Could not load image from {img1_path}")
        return
    if img2 is None:
        print(f"[Error] Could not load image from {img2_path}")
        return

    compare_images(detector, recognizer, img1, img2, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraFace Two-Photo Verification")
    parser.add_argument("--img1", type=str, default="", help="Path to first face photo")
    parser.add_argument("--img2", type=str, default="", help="Path to second face photo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index for interactive mode")
    parser.add_argument("--threshold", type=float, default=0.50, help="Matching similarity threshold (default: 0.50)")
    args = parser.parse_args()

    if args.img1 and args.img2:
        run_file_matching(args.img1, args.img2, threshold=args.threshold)
    else:
        run_interactive_matching(camera_index=args.camera, threshold=args.threshold)
