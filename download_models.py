import os
import urllib.request
from huggingface_hub import HfApi, hf_hub_download

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Download YuNet
yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
if not os.path.exists(yunet_path) or os.path.getsize(yunet_path) == 0:
    print("Downloading YuNet ONNX model...")
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(yunet_url, yunet_path)
    print(f"YuNet downloaded to {yunet_path}")
else:
    print(f"YuNet already exists at {yunet_path} (size: {os.path.getsize(yunet_path)} bytes)")

# 2. Inspect and download AuraFace files
print("Checking fal/AuraFace-v1 files...")
api = HfApi()
files = api.list_repo_files("fal/AuraFace-v1", token=False)
print("Files in fal/AuraFace-v1:", files)

for f in files:
    if f.endswith(".onnx") or f.endswith(".json") or f.endswith(".txt"):
        dest_file = os.path.join(MODELS_DIR, f)
        if not os.path.exists(dest_file):
            print(f"Downloading {f}...")
            hf_hub_download(
                repo_id="fal/AuraFace-v1",
                filename=f,
                local_dir=MODELS_DIR,
                token=False
            )
            print(f"Downloaded {f}")
        else:
            print(f"{f} already exists.")

print("All models ready!")
