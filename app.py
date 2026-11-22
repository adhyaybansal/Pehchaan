import os
import time
import threading
from typing import Optional, List, Dict
import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from core.face_detector import YuNetDetector, DetectedFace
from core.face_recognizer import AuraFaceRecognizer
from core.face_db import FaceDatabase
from core.unknown_manager import UnknownSnapsManager
from core.roi_manager import ROIManager
from core.relay_controller import RelayController

app = FastAPI(title="Face Recognition Door Access Control")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Core components
detector = YuNetDetector(score_threshold=0.40)
recognizer = AuraFaceRecognizer()
db = FaceDatabase(DATA_DIR)
unknown_mgr = UnknownSnapsManager(DATA_DIR)
roi_mgr = ROIManager()
relay = RelayController(DATA_DIR)

# Configuration state
MATCH_THRESHOLD = 0.50
camera_index = 0

# Latest system events for voice announcement & logs
latest_event = None
latest_frame_jpeg = None
frame_lock = threading.Lock()


def get_face_portrait(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(w * 0.25)
    pad_y = int(h * 0.25)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame, (180, 180))
    return cv2.resize(crop, (180, 180))


class CameraWorker:
    def __init__(self, cam_idx: int = 0):
        self.cam_idx = cam_idx
        self.running = False
        self.thread = None
        self.cap = None

    def _open_camera(self):
        # Connect to camera with DirectShow first (standard on Windows), then fallback
        for backend, name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Auto")]:
            try:
                cap = cv2.VideoCapture(self.cam_idx, backend)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[Camera] Connected to camera index {self.cam_idx} using {name} backend ({test_frame.shape[1]}x{test_frame.shape[0]})")
                        return cap
                    cap.release()
            except Exception:
                pass
        return None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        global latest_frame_jpeg, latest_event, MATCH_THRESHOLD
        try:
            self.cap = self._open_camera()
        except Exception as e:
            print(f"[Camera] Initialization error: {e}")
            self.cap = None

        use_simulated = (self.cap is None)
        if use_simulated:
            print("[Info] No active camera feed found. Operating in Simulation/Virtual Camera mode.")

        consecutive_failures = 0
        prev_time = time.time()
        fps = 0.0
        last_authorized_time = {}  # cooldown for re-unlocking per person
        active_authorized_until = 0.0  # grace period while authorized person is at door

        while self.running:
            frame = None
            if not use_simulated and self.cap is not None:
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    ret = False
                    frame = None

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print("[Warning] Camera feed lost or interrupted. Switching to Simulation mode.")
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                        use_simulated = True
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
            else:
                # Generate simulated camera canvas
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = (30, 25, 20)
                t_str = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, "SIMULATED CAMERA STREAM (Connect webcam to view live)", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
                cv2.putText(frame, t_str, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
                time.sleep(0.04)

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            fh, fw = frame.shape[:2]

            # 1. Apply ROI Crop
            crop, (ox, oy) = roi_mgr.crop(frame)

            # 2. Run YuNet Detection on cropped area
            detected_in_crop = detector.detect(crop)

            # 3. Process each face found
            vis_faces = []
            for face in detected_in_crop:
                # Map coords back to full frame
                fx, fy, fw_box, fh_box = face.bbox
                full_bbox = (fx + ox, fy + oy, fw_box, fh_box)
                full_landmarks = face.landmarks.copy()
                full_landmarks[:, 0] += ox
                full_landmarks[:, 1] += oy

                det_face = DetectedFace(bbox=full_bbox, landmarks=full_landmarks, score=face.score)
                vis_faces.append(det_face)

                # Extract embedding
                emb = recognizer.get_embedding(frame, full_landmarks)
                aligned_face = recognizer.align_face(frame, full_landmarks)

                # Match against registered database
                name, sim_score, pid = db.match(emb, threshold=MATCH_THRESHOLD)
                now = time.time()

                if name is not None:
                    # Authorized Person
                    if now - last_authorized_time.get(pid, 0) >= 5.0:
                        last_authorized_time[pid] = now
                        relay.unlock(person_name=name, confidence=sim_score)
                        latest_event = {
                            "id": f"{int(now * 1000)}_{pid}",
                            "name": name,
                            "status": "ALLOWED",
                            "score": sim_score
                        }
                else:
                    # Unrecognized face: if door is locked, capture to Unknowns
                    door_status = relay.get_status()
                    if not door_status["is_unlocked"]:
                        portrait_snap = get_face_portrait(frame, full_bbox)
                        unknown_mgr.record_unknown(
                            full_frame=frame,
                            face_crop=portrait_snap,
                            embedding=emb,
                            score=face.score
                        )

            # 4. Draw HUD on visualization frame
            vis_frame = roi_mgr.draw_roi(frame)
            vis_frame = detector.draw(vis_frame, vis_faces)

            # Overlay names / labels on faces
            for face in vis_faces:
                fx, fy, fw_box, fh_box = face.bbox
                emb = recognizer.get_embedding(frame, face.landmarks)
                name, sim_score, _ = db.match(emb, threshold=MATCH_THRESHOLD)
                if name:
                    tag_text = f"{name} ({int(sim_score * 100)}%)"
                    cv2.putText(vis_frame, tag_text, (fx, max(25, fy - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_frame, "Unknown Person", (fx, max(25, fy - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            # Overlay door state badge on video
            door_status = relay.get_status()
            if door_status["is_unlocked"]:
                door_str = f"DOOR UNLOCKED ({door_status['remaining_seconds']}s)"
                color = (0, 255, 0)
            else:
                door_str = "DOOR LOCKED"
                color = (0, 0, 255)

            cv2.putText(vis_frame, door_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (fw - 140, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with frame_lock:
                latest_frame_jpeg = buffer.tobytes()

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


cam_worker = CameraWorker(cam_idx=0)


@app.on_event("startup")
def startup_event():
    cam_worker.start()


@app.on_event("shutdown")
def shutdown_event():
    cam_worker.stop()


# Request Schemas
class TagRequest(BaseModel):
    snap_id: str
    name: str

class ROIRequest(BaseModel):
    points: List[List[int]]

class SettingsRequest(BaseModel):
    threshold: float
    esp32_url: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
def index_page():
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


def generate_mjpeg():
    while True:
        with frame_lock:
            frame_data = latest_frame_jpeg

        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.04)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    return {
        "door": relay.get_status(),
        "latest_event": latest_event,
        "match_threshold": MATCH_THRESHOLD
    }


@app.post("/api/door/unlock")
def manual_unlock():
    relay.unlock(person_name="Manual Admin", confidence=1.0)
    return {"status": "unlocked"}


@app.get("/api/unknowns")
def get_unknowns():
    return unknown_mgr.list_unknowns()


@app.post("/api/tag")
def tag_unknown(req: TagRequest):
    success = unknown_mgr.tag_person(req.snap_id, req.name, db)
    if not success:
        raise HTTPException(status_code=400, detail="Unknown snap not found")
    return {"status": "tagged", "name": req.name}


@app.delete("/api/unknowns/{snap_id}")
def dismiss_unknown(snap_id: str):
    success = unknown_mgr.delete_unknown(snap_id)
    return {"success": success}


@app.get("/api/people")
def get_people():
    return db.list_people()


@app.delete("/api/people/{person_id}")
def delete_person(person_id: str):
    db.delete_person(person_id)
    return {"status": "deleted"}


@app.get("/api/roi")
def get_roi():
    return {"points": roi_mgr.points}


@app.post("/api/roi")
def set_roi(req: ROIRequest):
    pts = [tuple(p) for p in req.points]
    roi_mgr.set_points(pts)
    return {"status": "saved", "points": roi_mgr.points}


@app.get("/api/logs")
def get_logs():
    return relay.logs


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    global MATCH_THRESHOLD
    MATCH_THRESHOLD = req.threshold
    relay.esp32_webhook_url = req.esp32_url
    return {"status": "saved", "threshold": MATCH_THRESHOLD}


@app.get("/audio/door_locked.mp3")
def get_door_locked_audio():
    audio_path = os.path.join(BASE_DIR, "static", "audio", "door_locked.mp3")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404)


@app.get("/audio/door_unlocked.mp3")
def get_door_unlocked_audio():
    audio_path = os.path.join(BASE_DIR, "static", "audio", "door_unlocked.mp3")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404)


@app.get("/snaps/{folder}/{filename}")
def serve_snap(folder: str, filename: str):
    if folder == "unknown":
        file_path = os.path.join(DATA_DIR, "unknown", filename)
    elif folder == "profiles":
        file_path = os.path.join(DATA_DIR, "profiles", filename)
    else:
        raise HTTPException(status_code=404)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404)
    return FileResponse(file_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face Recognition Door Access Control")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    args = parser.parse_args()

    cam_worker.cam_idx = args.camera
    uvicorn.run(app, host="0.0.0.0", port=args.port)
