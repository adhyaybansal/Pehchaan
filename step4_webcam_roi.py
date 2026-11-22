"""
Step 4: Interactive 4-Click ROI Selection & Restricted Detection

Usage:
  python step4_webcam_roi.py [--camera 0]

Controls:
  - Left Click (4 times): Mark the 4 corners of the active door detection zone.
  - 'r': Reset/Clear the selected points.
  - 's': Save current ROI to config/roi_config.json.
  - 'q': Quit.
"""

import cv2
import numpy as np
from core.face_detector import YuNetDetector, DetectedFace
from core.roi_manager import ROIManager

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Added point P{len(clicked_points)}: ({x}, {y})")
            if len(clicked_points) == 4:
                print("All 4 points selected! Detection will now run only within this ROI.")


def open_camera(cam_idx: int):
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


def run_roi_detection(camera_index: int = 0):
    global clicked_points
    detector = YuNetDetector(score_threshold=0.8)
    roi_mgr = ROIManager()

    # Preload existing points if saved
    clicked_points = list(roi_mgr.points)

    cap = open_camera(camera_index)
    if cap is None:
        print(f"[Error] Camera {camera_index} could not be opened.")
        return

    window_name = "Step 4: 4-Click ROI Face Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click 4 points on the window to define the detection zone.")
    print("2. Press 'r' to reset points.")
    print("3. Press 's' to save ROI.")
    print("4. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_mgr.points = clicked_points

        # Crop to ROI if 4 points are set
        crop, (ox, oy) = roi_mgr.crop(frame)

        # Run face detection on the crop
        detected_in_crop = detector.detect(crop)

        # Shift detected coordinates back to full frame
        full_faces = []
        for face in detected_in_crop:
            fx, fy, fw, fh = face.bbox
            full_bbox = (fx + ox, fy + oy, fw, fh)
            full_landmarks = face.landmarks.copy()
            full_landmarks[:, 0] += ox
            full_landmarks[:, 1] += oy
            full_faces.append(DetectedFace(
                bbox=full_bbox,
                landmarks=full_landmarks,
                score=face.score
            ))

        # Visual overlay
        vis = roi_mgr.draw_roi(frame)
        vis = detector.draw(vis, full_faces)

        # Status text
        status_text = f"ROI Points: {len(clicked_points)}/4 | Faces in ROI: {len(full_faces)}"
        cv2.putText(vis, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        help_text = "[Click 4 pts] | 'r': Reset | 's': Save | 'q': Quit"
        cv2.putText(vis, help_text, (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            clicked_points = []
            roi_mgr.set_points([])
            print("ROI points reset.")
        elif key == ord('s'):
            roi_mgr.set_points(clicked_points)
            print("ROI points saved successfully to config/roi_config.json!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 4: 4-Click ROI Face Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    run_roi_detection(camera_index=args.camera)
