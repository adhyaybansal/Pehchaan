import os
import json
from typing import List, Tuple, Optional
import cv2
import numpy as np


class ROIManager:
    """
    Manages the 4-point Region of Interest (ROI) selection and cropping.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "roi_config.json")

        self.config_path = config_path
        self.points: List[Tuple[int, int]] = []
        self.load()

    def set_points(self, points: List[Tuple[int, int]]):
        """Set the 4 points defining the ROI (automatically sorted clockwise)."""
        if len(points) == 4:
            import math
            cx = sum(p[0] for p in points) / 4.0
            cy = sum(p[1] for p in points) / 4.0
            points = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        self.points = points
        self.save()

    def get_bounding_rect(self, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (x, y, w, h) covering all ROI points, clipped to frame bounds.
        If fewer than 4 points exist, returns None (full frame).
        """
        if len(self.points) < 4:
            return None

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        min_x = max(0, min(xs))
        min_y = max(0, min(ys))
        max_x = min(frame_w, max(xs))
        max_y = min(frame_h, max(ys))

        w = max(0, max_x - min_x)
        h = max(0, max_y - min_y)

        if w == 0 or h == 0:
            return None

        return (min_x, min_y, w, h)

    def crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crops frame to the bounding box of ROI.
        Returns:
            cropped_frame: np.ndarray
            offset: (offset_x, offset_y) to map coordinates back to full frame
        """
        fh, fw = frame.shape[:2]
        rect = self.get_bounding_rect(fw, fh)
        if rect is None:
            return frame, (0, 0)

        x, y, w, h = rect
        return frame[y:y+h, x:x+w], (x, y)

    def draw_roi(self, frame: np.ndarray, color=(255, 200, 0), thickness=2) -> np.ndarray:
        """Draws the 4-point polygon and bounding box overlay on frame."""
        vis = frame.copy()
        if len(self.points) == 4:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

            for idx, pt in enumerate(self.points):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # Draw the enclosing crop rectangle in thin green
            fh, fw = frame.shape[:2]
            rect = self.get_bounding_rect(fw, fh)
            if rect:
                rx, ry, rw, rh = rect
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
                cv2.putText(vis, "ROI Zone", (rx, max(18, ry - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        return vis

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump({"points": self.points}, f, indent=2)

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.points = [tuple(p) for p in data.get("points", [])]
            except Exception as e:
                print(f"[Warning] Failed to load ROI config: {e}")
                self.points = []
