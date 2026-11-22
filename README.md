# Pehchaan

A local, offline-first facial recognition door access control system designed to run on-premise without cloud dependencies. The system detects approaching visitors, identifies authorized individuals using 512-dimensional vector embeddings, announces access grants, and manages door lock relays with automatic relock timers.

---

## Technical Architecture

The recognition pipeline is built entirely around local inference:

```
[ Camera Stream (DirectShow / USB / RTSP) ]
                   │
                   ▼
[ 4-Point Region of Interest (ROI) ]
   • Bounded entry area defined by an administrator
   • Restricts face detection to the doorway, ignoring background movement
                   │
                   ▼
[ Contrast Normalization (CLAHE) ]
   • Equalizes luminance shadows caused by backlighting or overhead lights
                   │
                   ▼
[ Face Detection & Alignment (YuNet) ]
   • Extracts bounding box and 5 facial landmarks (eyes, nose, mouth corners)
   • Calculates affine similarity transformation to level and crop the face to 112x112
                   │
                   ▼
[ Feature Extraction (AuraFace - ArcFace ResNet100) ]
   • Converts aligned 112x112 crop into a normalized 512-dimensional vector
                   │
                   ▼
[ Vector Matching (Cosine Similarity) ]
   • Compares vector against enrolled profiles (up to 5 embeddings per person)
   • Multi-sample profile matching takes the highest similarity across angles
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Match >= 0.50 ]     [ Match < 0.50 ]
 Authorized Access      Unknown Visitor
 • Door unlocks (5s)    • Saves portrait to review queue
 • Speech announcement  • Time-bounded deduplication (20s)
 • Entry logged         • Admin tags with name in dashboard
```

---

## Core Features

- **Local Inference:** Runs completely offline on standard CPU hardware using OpenCV and ONNX Runtime. No cloud APIs, no external telemetry, and biometric data remains entirely on-device.
- **Commercially Permissive Weights:** Uses AuraFace (`glintr100.onnx`) and YuNet, which are free for commercial deployment (avoiding non-commercial licensing constraints found in InsightFace default bundles).
- **Interactive 4-Point ROI Calibration:** Admins can click 4 points directly on the live video stream in the web interface to define the active door area. Detection runs only inside this zone.
- **Multi-Angle Enrollment:** Profiles store up to 5 face embeddings per person. When an operator tags multiple snapshots of the same person from different angles or lighting conditions, the profile automatically accumulates samples to maximize reliability.
- **Visitor Queue with Deduplication:** Unrecognized visitors are captured and presented in an "Unknowns" tab for review. An embedding deduplication filter prevents continuous photo capture while someone is standing at the door.
- **Audio Feedback & Speech Synthesis:** Announces the authorized person's name aloud via the browser speech engine on entry, with dedicated audio cues when the door locks.
- **Hardware Integration:** Manages a 5-second unlock duration with an auto-relock loop, and provides an HTTP webhook trigger for network-connected microcontrollers (ESP32, Raspberry Pi, Shelly relays).

---

## Project Structure

```
Pehchaan/
├── app.py                      # FastAPI web server and camera processing loop
├── download_models.py          # Automatic downloader for YuNet and AuraFace ONNX models
├── requirements.txt            # Python dependencies
├── core/
│   ├── face_detector.py        # YuNet detector with CLAHE contrast normalization
│   ├── face_recognizer.py      # 5-point affine alignment and AuraFace embedding extraction
│   ├── face_db.py              # Profile storage and multi-embedding cosine matching
│   ├── unknown_manager.py      # Unrecognized visitor capture and review queue
│   ├── roi_manager.py          # 4-point quadrilateral cropping and coordinate re-projection
│   └── relay_controller.py     # Door state management, timers, and access audit logs
├── templates/
│   └── index.html              # Responsive web dashboard (live stream, tagging, logs, settings)
├── static/
│   └── audio/                  # Audio cues for door lock/unlock states
├── step2_webcam_detection.py   # Standalone CLI test for YuNet face detection
├── step3_test_matching.py      # Standalone CLI test for two-photo face verification
└── step4_webcam_roi.py         # Standalone CLI test for 4-click ROI selection
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git
- A connected USB webcam or integrated laptop camera

### 2. Installation

Clone the repository and enter the directory:
```bash
git clone https://github.com/adhyaybansal/Pehchaan.git
cd Pehchaan
```

Create and activate a virtual environment:

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the pre-trained ONNX models:
```bash
python download_models.py
```
This fetches the YuNet detector (`face_detection_yunet_2023mar.onnx`) and AuraFace embedder (`glintr100.onnx`) directly into the `models/` directory.

---

## Running the Application

Start the web dashboard:
```bash
python app.py
```

If your machine has multiple video devices (for example, an integrated webcam and an external USB camera), you can specify the camera index:
```bash
python app.py --camera 1
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## Operating Workflow

1. **Set the Door Area (ROI):** Click 4 corners on the live camera stream in the dashboard to define the active detection area, then click **Apply ROI**.
2. **Review Unknown Visitors:** Stand in front of the camera. The system will detect your face, verify that no matching profile exists, and save a portrait snapshot in the **Unknowns** tab.
3. **Enroll a Profile:** In the **Unknowns** tab, click your snapshot, enter your name, and click **Tag & Authorize**.
4. **Multi-Angle Enrichment:** To make recognition more robust, take a snapshot with your head slightly turned, click **Tag**, select your existing name from the dropdown or pill buttons, and save. The profile will now recognize both angles.
5. **Access Grant:** Look at the camera. The system will identify your face, display your name with match percentage, unlock the door for 5 seconds, and announce your name aloud.

---

## Standalone Diagnostic Scripts

If you want to test individual components without starting the full web server:

- **Test YuNet Detection:**
  ```bash
  python step2_webcam_detection.py
  ```
- **Test Two-Photo Vector Matching:**
  ```bash
  python step3_test_matching.py --img1 person_photo_1.jpg --img2 person_photo_2.jpg
  ```
- **Test Interactive ROI Selection:**
  ```bash
  python step4_webcam_roi.py
  ```

---

## Hardware Relay Configuration

To trigger a physical door lock relay:
1. Navigate to the **Settings** tab in the dashboard.
2. Enter your microcontroller's webhook URL (e.g. `http://192.168.1.50/relay/open`).
3. Click **Save Settings**.
When an authorized face is recognized or manual unlock is clicked, an asynchronous HTTP GET request is dispatched to trigger the physical relay without blocking the video pipeline.
