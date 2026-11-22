"""
Step 2: Detect faces from Webcam using OpenCV YuNet
Press 'q' to quit.
Press 's' to save a snapshot of the current frame into the 'snapshots' folder.
"""

import os
import time
import cv2
from core.face_detector import YuNetDetector


def open_camera(cam_idx: int):
    # Try DirectShow first on Windows (avoids MSMF -1072873821 error)
    for backend, name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "Auto")]:
        try:
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[Camera] Connected using {name} backend ({test_frame.shape[1]}x{test_frame.shape[0]})")
                    return cap
                cap.release()
        except Exception:
            pass
    return None


def run_webcam_detection(camera_index: int = 0):
    detector = YuNetDetector(score_threshold=0.8)

    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    print(f"Opening camera index {camera_index}...")
    cap = open_camera(camera_index)

    if cap is None:
        print(f"[Error] Could not open camera {camera_index}. Check if webcam is connected, or try another index (e.g. 1).")
        return

    print("Camera successfully opened!")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot to 'snapshots/' folder")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame from camera. Exiting...")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        # Run detection
        faces = detector.detect(frame)

        # Draw detections
        vis_frame = detector.draw(frame, faces)

        # Overlay info
        info_text = f"Faces: {len(faces)} | FPS: {fps:.1f}"
        cv2.putText(vis_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show live window
        cv2.imshow("Step 2: YuNet Face Detection (Press 'q' to quit, 's' to save)", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting webcam detection...")
            break
        elif key == ord('s'):
            snap_time = int(time.time())
            snap_path = os.path.join(snapshots_dir, f"snap_{snap_time}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[Snapshot] Saved frame to {snap_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YuNet Webcam Face Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    run_webcam_detection(camera_index=args.camera)
