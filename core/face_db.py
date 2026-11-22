import os
import json
import time
import uuid
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


class FaceDatabase:
    """
    Local face database storing enrolled persons and their 512-dim AuraFace embeddings.
    Supports up to 5 embeddings per person for multi-angle/multi-lighting accuracy.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")

        self.data_dir = data_dir
        self.profiles_dir = os.path.join(data_dir, "profiles")
        self.db_meta_file = os.path.join(data_dir, "db.json")
        self.embeddings_file = os.path.join(data_dir, "embeddings.npz")

        os.makedirs(self.profiles_dir, exist_ok=True)

        # In-memory index:
        # self.people = {person_id: {"name": str, "created_at": float, "snap_filenames": [...]}}
        # self.embeddings = {person_id: np.ndarray shape (N, 512)}
        self.people: Dict[str, dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

        self.load()

    def add_person(self, name: str, embedding: np.ndarray, face_crop: Optional[np.ndarray] = None) -> str:
        """
        Enrolls a new person or adds an embedding snap to an existing person with the same name.
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        person_id = None
        # Check if name already exists
        for pid, pdata in self.people.items():
            if pdata["name"].strip().lower() == name.strip().lower():
                person_id = pid
                break

        now = time.time()
        snap_filename = None

        if person_id is None:
            person_id = str(uuid.uuid4())[:8]
            self.people[person_id] = {
                "id": person_id,
                "name": name.strip(),
                "created_at": now,
                "snap_filenames": []
            }
            self.embeddings[person_id] = np.expand_dims(embedding, axis=0)
        else:
            # Append embedding (keep up to 5 best/latest)
            curr_embs = self.embeddings[person_id]
            if len(curr_embs) >= 5:
                curr_embs = curr_embs[1:]  # drop oldest
            self.embeddings[person_id] = np.vstack([curr_embs, embedding])

        if face_crop is not None:
            snap_filename = f"{person_id}_{int(now)}_{len(self.people[person_id]['snap_filenames'])+1}.jpg"
            snap_path = os.path.join(self.profiles_dir, snap_filename)
            cv2.imwrite(snap_path, face_crop)
            self.people[person_id]["snap_filenames"].append(snap_filename)

        self.save()
        return person_id

    def match(self, query_embedding: np.ndarray, threshold: float = 0.50) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Matches a query embedding against the database.
        Returns:
            (person_name, similarity_score, person_id)
            If no match meets threshold: (None, best_score, None)
        """
        if len(self.embeddings) == 0:
            return None, 0.0, None

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        best_person_id = None
        best_name = None
        best_sim = -1.0

        for pid, embs in self.embeddings.items():
            # Dot product against all stored embeddings for this person
            sims = np.dot(embs, query_embedding)
            max_person_sim = float(np.max(sims))
            if max_person_sim > best_sim:
                best_sim = max_person_sim
                best_person_id = pid
                best_name = self.people[pid]["name"]

        if best_sim >= threshold:
            return best_name, best_sim, best_person_id
        else:
            return None, best_sim, None

    def list_people(self) -> List[dict]:
        return list(self.people.values())

    def delete_person(self, person_id: str):
        if person_id in self.people:
            # Remove snaps
            for sf in self.people[person_id].get("snap_filenames", []):
                p = os.path.join(self.profiles_dir, sf)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            del self.people[person_id]
        if person_id in self.embeddings:
            del self.embeddings[person_id]
        self.save()

    def save(self):
        with open(self.db_meta_file, "w") as f:
            json.dump(self.people, f, indent=2)

        # Save embeddings
        np.savez_compressed(self.embeddings_file, **self.embeddings)

    def load(self):
        if os.path.exists(self.db_meta_file):
            try:
                with open(self.db_meta_file, "r") as f:
                    self.people = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load db.json: {e}")
                self.people = {}

        if os.path.exists(self.embeddings_file):
            try:
                npz = np.load(self.embeddings_file)
                self.embeddings = {k: npz[k] for k in npz.files}
            except Exception as e:
                print(f"[Warning] Failed to load embeddings.npz: {e}")
                self.embeddings = {}
