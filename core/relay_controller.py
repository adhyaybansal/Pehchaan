import os
import json
import time
import threading
from typing import Optional, Dict
import requests


class RelayController:
    """
    Door Unlock Relay Controller.
    Supports Simulation, HTTP Webhook (ESP32/Shelly/Tasmota), and Raspberry Pi GPIO.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")

        self.data_dir = data_dir
        self.logs_file = os.path.join(data_dir, "access_logs.json")
        os.makedirs(data_dir, exist_ok=True)

        self.is_unlocked = False
        self.unlock_end_time = 0.0
        self.lock_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Configurable settings
        self.unlock_duration = 5.0  # seconds
        self.esp32_webhook_url = ""  # e.g., "http://192.168.1.50/unlock"
        self.gpio_pin: Optional[int] = None  # for Raspberry Pi

        self.logs = []
        self.load_logs()

    def unlock(self, person_name: str, confidence: float, snap_path: Optional[str] = None):
        """
        Triggers door unlock for `self.unlock_duration` seconds and logs the entry.
        """
        now = time.time()
        with self._lock:
            self.is_unlocked = True
            self.unlock_end_time = now + self.unlock_duration

            # Log event
            log_entry = {
                "id": len(self.logs) + 1,
                "timestamp": now,
                "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "name": person_name,
                "status": "ALLOWED",
                "confidence": round(confidence, 3),
                "snap_path": snap_path or ""
            }
            self.logs.insert(0, log_entry)
            if len(self.logs) > 500:
                self.logs = self.logs[:500]
            self.save_logs()

            # Hardware trigger in a separate thread so video stream never blocks
            threading.Thread(target=self._hardware_trigger, daemon=True).start()

    def log_denied(self, confidence: float, snap_path: Optional[str] = None):
        now = time.time()
        with self._lock:
            log_entry = {
                "id": len(self.logs) + 1,
                "timestamp": now,
                "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "name": "Unknown",
                "status": "DENIED",
                "confidence": round(confidence, 3),
                "snap_path": snap_path or ""
            }
            self.logs.insert(0, log_entry)
            if len(self.logs) > 500:
                self.logs = self.logs[:500]
            self.save_logs()

    def get_status(self) -> Dict:
        with self._lock:
            remaining = max(0.0, self.unlock_end_time - time.time())
            if remaining == 0.0:
                self.is_unlocked = False

            return {
                "is_unlocked": self.is_unlocked,
                "remaining_seconds": round(remaining, 1),
                "door_state": "UNLOCKED" if self.is_unlocked else "LOCKED"
            }

    def _hardware_trigger(self):
        """Sends HTTP request to ESP32 relay if configured."""
        if self.esp32_webhook_url:
            try:
                requests.get(self.esp32_webhook_url, timeout=2.0)
            except Exception as e:
                print(f"[Relay] Webhook error: {e}")

    def load_logs(self):
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, "r") as f:
                    self.logs = json.load(f)
            except Exception:
                self.logs = []

    def save_logs(self):
        try:
            with open(self.logs_file, "w") as f:
                json.dump(self.logs, f, indent=2)
        except Exception:
            pass
