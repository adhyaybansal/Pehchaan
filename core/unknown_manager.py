import os
import json
import time
import uuid
from typing import List, Optional, Dict
import cv2
import numpy as np
from core.face_db import FaceDatabase


class UnknownSnapsManager:
    """
    Manages capturing, listing, and tagging unknown faces.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")

        self.data_dir = data_dir
        self.unknown_dir = os.path.join(data_dir, "unknown")
        self.meta_file = os.path.join(data_dir, "unknown_meta.json")
        self.embeddings_file = os.path.join(data_dir, "unknown_embeddings.npz")

        os.makedirs(self.unknown_dir, exist_ok=True)
        self.snaps: Dict[str, dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.load()

        # Cooldown dictionary to avoid saving 30 photos a second of the same unknown person standing there
        # {face_hash/id: last_captured_time}
        self.last_capture_time = 0.0
        self.capture_cooldown_seconds = 3.0

    def record_unknown(
        self,
        full_frame: np.ndarray,
        face_crop: np.ndarray,
        embedding: np.ndarray,
        score: float
    ) -> Optional[str]:
        """
        Records an unknown face detection if cooldown has passed and face isn't already recently queued.
        Returns the snap_id if recorded, or None if throttled/duplicate.
        """
        now = time.time()
        # Cooldown check
        if now - self.last_capture_time < 2.5:
            return None

        # Deduplication: do not capture duplicate if the same face was already captured in the last 20 seconds
        norm_emb = embedding / max(1e-6, np.linalg.norm(embedding))
        for snap_id, meta in list(self.snaps.items()):
            if now - meta.get("timestamp", 0) < 20.0:
                existing_emb = self.embeddings.get(snap_id)
                if existing_emb is not None:
                    existing_norm = existing_emb / max(1e-6, np.linalg.norm(existing_emb))
                    sim = float(np.dot(norm_emb, existing_norm))
                    if sim >= 0.55:
                        return None

        self.last_capture_time = now
        snap_id = str(uuid.uuid4())[:8]

        # Save crops
        crop_filename = f"crop_{snap_id}.jpg"
        full_filename = f"full_{snap_id}.jpg"
        cv2.imwrite(os.path.join(self.unknown_dir, crop_filename), face_crop)
        cv2.imwrite(os.path.join(self.unknown_dir, full_filename), full_frame)

        self.snaps[snap_id] = {
            "id": snap_id,
            "timestamp": now,
            "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "crop_filename": crop_filename,
            "full_filename": full_filename,
            "score": float(score)
        }
        self.embeddings[snap_id] = embedding

        self.save()
        return snap_id

    def list_unknowns(self) -> List[dict]:
        """Returns sorted list of unknown snaps (newest first)."""
        items = list(self.snaps.values())
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items

    def tag_person(self, snap_id: str, person_name: str, db: FaceDatabase) -> bool:
        """
        Tags an unknown face with a name, enrolling them into the FaceDatabase,
        and deletes the item from unknown queue.
        """
        if snap_id not in self.snaps or snap_id not in self.embeddings:
            return False

        meta = self.snaps[snap_id]
        embedding = self.embeddings[snap_id]
        crop_path = os.path.join(self.unknown_dir, meta["crop_filename"])

        crop_img = None
        if os.path.exists(crop_path):
            crop_img = cv2.imread(crop_path)

        # Enroll in FaceDatabase
        db.add_person(name=person_name, embedding=embedding, face_crop=crop_img)

        # Cleanup unknown files
        full_path = os.path.join(self.unknown_dir, meta["full_filename"])
        for p in [crop_path, full_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        del self.snaps[snap_id]
        del self.embeddings[snap_id]
        self.save()
        return True

    def delete_unknown(self, snap_id: str) -> bool:
        if snap_id not in self.snaps:
            return False

        meta = self.snaps[snap_id]
        crop_path = os.path.join(self.unknown_dir, meta.get("crop_filename", ""))
        full_path = os.path.join(self.unknown_dir, meta.get("full_filename", ""))
        for p in [crop_path, full_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        del self.snaps[snap_id]
        if snap_id in self.embeddings:
            del self.embeddings[snap_id]
        self.save()
        return True

    def save(self):
        with open(self.meta_file, "w") as f:
            json.dump(self.snaps, f, indent=2)
        np.savez_compressed(self.embeddings_file, **self.embeddings)

    def load(self):
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, "r") as f:
                    self.snaps = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load unknown_meta.json: {e}")
                self.snaps = {}

        if os.path.exists(self.embeddings_file):
            try:
                npz = np.load(self.embeddings_file)
                self.embeddings = {k: npz[k] for k in npz.files}
            except Exception as e:
                print(f"[Warning] Failed to load unknown_embeddings.npz: {e}")
                self.embeddings = {}
