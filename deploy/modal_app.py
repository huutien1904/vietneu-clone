"""
Deploy VieNeu-TTS API lên Modal.com (serverless, CPU, trả tiền theo giây chạy).
=================================================================================
Gói ``apps/api_server.py`` + model v3 Turbo (ONNX int8, tải sẵn vào image) + giọng
người dùng (clone_A, ...) thành một web endpoint HTTPS. Container tự tắt khi rảnh
(``scaledown_window``) nên chi phí chỉ phát sinh khi có request.

Cài & đăng nhập (một lần, dùng tài khoản GitHub):
    pip install modal
    modal setup

Deploy (chạy từ thư mục gốc repo, nơi có src/ và apps/):
    set VIENEU_API_KEY=chuoi-bi-mat        # PowerShell: $env:VIENEU_API_KEY="chuoi-bi-mat"
    modal deploy deploy/modal_app.py
    → in ra URL dạng https://<workspace>--vieneu-tts.modal.run

Thử nhanh:
    curl -X POST https://<url>/tts -H "X-API-Key: chuoi-bi-mat" -H "Content-Type: application/json" \
         -d '{"text": "Xin chào", "voice": "clone_A"}' -o out.wav

Tuỳ chỉnh bằng biến môi trường lúc deploy:
    MODAL_CPU=2 MODAL_MEMORY_MB=4096   tài nguyên container. Modal tính `cpu` theo NHÂN VẬT LÝ (1 = 2 vCPU),
                                       nên 2 = 4 vCPU (~0,13 USD/giờ chạy kèm 4 GiB); dùng 4 cho văn bản dài
    MODAL_SCALEDOWN_SECONDS=120        giữ container ấm bao lâu sau request cuối (2 s đến 20 phút; lúc ấm vẫn tính tiền)
    MODAL_MEMORY_SNAPSHOT=1            thử bật memory snapshot để giảm cold start (xem docs/DEPLOY_MODAL.md)
    VIENEU_API_KEY                     API key bắt buộc cho client (nên đặt: URL là public)

Giọng người dùng: file ``~/.cache/vieneu/user_voices.json`` trên máy deploy được nhúng làm
"hạt giống" và chép vào Volume ``vieneu-voices`` ở lần chạy đầu. Giọng thêm qua
``POST /voices`` trên cloud cũng được lưu vào Volume nên không mất khi container tắt.
"""
import os
import shutil
from pathlib import Path

import modal

APP_NAME = "vieneu-tts"
REPO = Path(__file__).resolve().parents[1]

CPU = float(os.environ.get("MODAL_CPU", "2"))          # nhân vật lý (Modal: 1 core = 2 vCPU)
MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "4096"))
THREADS = max(1, int(round(CPU * 2)))                    # luồng ONNX = số vCPU được cấp
SCALEDOWN_SECONDS = int(os.environ.get("MODAL_SCALEDOWN_SECONDS", "120"))
USE_SNAPSHOT = os.environ.get("MODAL_MEMORY_SNAPSHOT", "0") == "1"

VOLUME_NAME = "vieneu-voices"
VOLUME_DIR = "/data"                                   # Volume: giọng người dùng (bền)
USER_VOICES_PATH = f"{VOLUME_DIR}/user_voices.json"
SEED_VOICES_LOCAL = Path.home() / ".cache" / "vieneu" / "user_voices.json"
SEED_VOICES_REMOTE = "/root/app/seed_user_voices.json"

# Các gói core torch-free của vieneu (khớp pyproject [project] dependencies, bỏ gradio/PyYAML
# vì API server không dùng) + FastAPI stack.
PY_DEPS = [
    "sea-g2p>=0.9.0", "onnxruntime>=1.20.0", "numpy", "soundfile", "soxr",
    "kaldi-native-fbank>=1.20", "tokenizers>=0.20", "huggingface_hub", "librosa>=0.11.0",
    "fastapi", "uvicorn", "python-multipart",
]


def download_models() -> None:
    """Tải sẵn model vào image lúc build → runtime chạy HF_HUB_OFFLINE=1, không gọi mạng."""
    from huggingface_hub import snapshot_download
    snapshot_download("pnnbao-ump/VieNeu-TTS-v3-Turbo",
                      allow_patterns=["config.json", "denoiser.onnx", "speaker_encoder.onnx", "onnx_int8/*"])
    snapshot_download("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(*PY_DEPS)
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"})
    .run_function(download_models)
    .env({
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": "/root/app/src:/root/app",
        "VIENEU_USER_VOICES": USER_VOICES_PATH,
        "VIENEU_THREADS": str(THREADS),
    })
    .add_local_dir(REPO / "src", "/root/app/src", ignore=["**/__pycache__", "**/*.egg-info"])
    .add_local_dir(REPO / "apps", "/root/app/apps", ignore=["**/__pycache__"])
)
if SEED_VOICES_LOCAL.exists():
    image = image.add_local_file(SEED_VOICES_LOCAL, SEED_VOICES_REMOTE)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# API key lấy từ môi trường máy deploy. Muốn quản lý qua Modal Secrets:
#   modal secret create vieneu-api VIENEU_API_KEY=...   rồi thay bằng modal.Secret.from_name("vieneu-api")
secret = modal.Secret.from_dict({"VIENEU_API_KEY": os.environ.get("VIENEU_API_KEY", "")})

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    cpu=CPU,
    memory=MEMORY_MB,
    volumes={VOLUME_DIR: volume},
    secrets=[secret],
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=600,             # mỗi input; lưu ý web request qua modal.run vẫn bị cắt ở 150 s (xem docs/DEPLOY_MODAL.md)
    startup_timeout=300,     # nạp model lúc khởi động
    min_containers=0,        # scale về 0 khi rảnh → không tốn credit
    enable_memory_snapshot=USE_SNAPSHOT,
)
@modal.concurrent(max_inputs=16)   # nhiều request vào cùng container; api_server tự xếp hàng engine
class TTSService:
    @modal.enter(snap=USE_SNAPSHOT)
    def load(self) -> None:
        # 1) Hạt giống giọng người dùng → Volume (chỉ lần đầu, khi Volume chưa có file)
        dst = Path(USER_VOICES_PATH)
        if not dst.exists() and Path(SEED_VOICES_REMOTE).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SEED_VOICES_REMOTE, dst)
            volume.commit()

        # 2) Nạp model một lần cho cả vòng đời container
        from apps.api_server import app as api, load_tts
        api.state.threads = THREADS
        api.state.api_key = os.environ.get("VIENEU_API_KEY", "").strip()
        api.state.tts = load_tts(threads=THREADS)

        # 3) Ghi Volume sau mỗi thay đổi giọng để giọng mới không mất khi container tắt
        @api.middleware("http")
        async def _persist_voices(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/voices") and request.method in ("POST", "DELETE") \
                    and response.status_code < 300:
                volume.commit()
            return response

        self.api = api
        n = len(api.state.tts.list_voices())
        print(f"✅ VieNeu-TTS sẵn sàng trên Modal: {n} giọng, threads={THREADS}, "
              f"auth={'bật' if api.state.api_key else 'TẮT'}", flush=True)

    @modal.asgi_app(label="vieneu-tts")   # URL: https://<workspace>--vieneu-tts.modal.run
    def web(self):
        return self.api
