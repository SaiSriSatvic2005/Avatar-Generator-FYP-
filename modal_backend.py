import os
import sys
import modal

# -----------------------------------------------------------------------------
# 1. Cloud Function to Download Heavy 3D Models inside Datacenter (10 Gbps)
# -----------------------------------------------------------------------------
def download_model_checkpoints():
    import os
    import urllib.request
    from huggingface_hub import hf_hub_download

    print("[Modal Build] Fast-downloading HaMeR 3D Checkpoint from Hugging Face...")
    ckpt_dir = "/root/hamer/_DATA/hamer_ckpts/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    hf_hub_download(
        repo_id="Leo-TX/hamer",
        filename="hamer.ckpt",
        local_dir=ckpt_dir
    )

    print("[Modal Build] Downloading MANO Hand Mesh Assets...")
    mano_dir = "/root/hamer/_DATA/data/mano"
    os.makedirs(mano_dir, exist_ok=True)
    mano_right = os.path.join(mano_dir, "MANO_RIGHT.pkl")
    if not os.path.exists(mano_right):
        urllib.request.urlretrieve("https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl", mano_right)

    print("[Modal Build] Downloading MediaPipe Hand Landmarker Model...")
    mp_task = "/root/hamer/hand_landmarker.task"
    if not os.path.exists(mp_task):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            mp_task
        )
    print("[Modal Build] All 3D foundation assets cached in container!")

# -----------------------------------------------------------------------------
# 2. Build Container Image with System Libraries & Pip Dependencies
# -----------------------------------------------------------------------------
app = modal.App("sign-avatar-backend")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libgles2-mesa",          # Provides libGLESv2.so.2 required by MediaPipe
        "libegl1-mesa",           # Provides libEGL.so.1 required by MediaPipe
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "git",
        "wget",
        "ffmpeg"
    )
    .pip_install(
        "torch",
        "torchvision",
        "mediapipe>=0.10.9,<0.11.0",
        "opencv-python-headless",
        "numpy<2.0.0",
        "scikit-learn",
        "scikit-image",
        "joblib",
        "scipy",
        "pandas",
        "Flask>=3.0.0",
        "fastapi",
        "python-multipart",
        "timm",
        "einops",
        "smplx==0.1.28",
        "chumpy",
        "pytorch-lightning",
        "yacs",
        "huggingface_hub"
    )
    # Upload lightweight local code and configurations first (~25 MB total)
    .add_local_dir(
        "hamer",
        remote_path="/root/hamer",
        copy=True,
        ignore=[
            "**/vitpose_ckpts/**",
            "**/*.zip",
            "**/*.tar.gz",
            "**/.git/**",
            "**/*.task",
            "**/*.pkl",
            "**/*.ckpt",
            "**/*.pth"
        ]
    )
    .add_local_file("hamer/_DATA/hamer_ckpts/model_config.yaml", remote_path="/root/hamer/_DATA/hamer_ckpts/model_config.yaml", copy=True)
    .add_local_file("hamer/_DATA/hamer_ckpts/dataset_config.yaml", remote_path="/root/hamer/_DATA/hamer_ckpts/dataset_config.yaml", copy=True)
    .add_local_file("hamer/_DATA/data/mano_mean_params.npz", remote_path="/root/hamer/_DATA/data/mano_mean_params.npz", copy=True)
    .add_local_dir(
        "Integration-20260706T062240Z-3-001",
        remote_path="/root/Integration-20260706T062240Z-3-001",
        copy=True,
        ignore=[
            "**/ISL_sign_videos/**",
            "**/*.MOV",
            "**/*.mov",
            "**/*.mp4"
        ]
    )
    .add_local_dir("Senior Code", remote_path="/root/Senior Code", copy=True)
    .add_local_dir("webapp", remote_path="/root/webapp", copy=True, ignore=["uploads/**", "**/__pycache__/**"])
    .add_local_dir("downloaded_dgs_dataset", remote_path="/root/downloaded_dgs_dataset", copy=True)
    .add_local_file("ISL_videos/Hello.mp4", remote_path="/root/ISL_videos/Hello.mp4", copy=True)
    .add_local_file("ISL_videos/Welcome.mp4", remote_path="/root/ISL_videos/Welcome.mp4", copy=True)
    .add_local_file("local_hamer_pipeline.py", remote_path="/root/local_hamer_pipeline.py", copy=True)
    # Download heavy 2.5 GB model checkpoints directly inside the cloud datacenter (10 Gbps)
    .run_function(download_model_checkpoints)
)
# -----------------------------------------------------------------------------
# 3. Expose Native ASGI Web Application with NVIDIA T4 GPU Acceleration
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",                 # Fast NVIDIA T4 GPU
    cpu=4.0,                  # 4 Dedicated vCPUs for video decoding and MediaPipe tracking
    memory=8192,              # 8 GB RAM (prevent OOM termination during heavy 3D ViT inference)
    max_containers=2,         # Limit active GPU containers
    scaledown_window=600,     # Keep warm for 10 minutes after last request
    timeout=600               # Max 10 min per request (video processing can be slow)
)
@modal.asgi_app()
def flask_app():
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/hamer")
    sys.path.insert(0, "/root/Integration-20260706T062240Z-3-001/Integration")
    
    from webapp.app import get_pipeline
    from webapp.asgi_app import app as asgi_web_app

    # Preload all ML models into GPU VRAM at container boot time.
    # This runs ONCE when the container starts, NOT on every request.
    print("[Modal Boot] Preloading V8 Hybrid Pipeline (HaMeR 3D + ResNet + BiGRU)...")
    get_pipeline()
    print("[Modal Boot] All models loaded into GPU VRAM. Ready for fast inference!")

    return asgi_web_app


