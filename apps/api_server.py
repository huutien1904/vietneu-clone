"""
VieNeu-TTS HTTP API (FastAPI)
=============================
API để ứng dụng khác gọi TTS bằng giọng có sẵn hoặc giọng đã clone (giọng người
dùng lưu bền trong ``~/.cache/vieneu/user_voices.json`` hoặc ``$VIENEU_USER_VOICES``).

    uv run --frozen python -m apps.api_server                 # http://127.0.0.1:8000/docs
    uv run --frozen vieneu-api --host 0.0.0.0 --port 8000     # mở ra mạng (nên đặt VIENEU_API_KEY)

Endpoints
    GET    /health                 trạng thái, backend, số giọng (không cần key)
    GET    /voices                 danh sách giọng (cờ ``custom`` = giọng người dùng)
    POST   /voices   (multipart)   thêm giọng clone từ file audio 3-8 s và lưu bền
    DELETE /voices/{name}          xoá giọng người dùng (giọng có sẵn không xoá được)
    POST   /tts                    JSON → audio (wav/pcm/flac; mp3/opus/aac nếu có ffmpeg)
    POST   /tts/stream             JSON → WAV stream theo chunk (nghe ngay sau ~300 ms)
    POST   /v1/audio/speech        tương thích OpenAI Audio API (``input``, ``voice``, ``response_format``)
    GET    /v1/models              tương thích OpenAI (một model: ``vieneu-v3-turbo``)

Biến môi trường
    VIENEU_API_HOST / VIENEU_API_PORT   mặc định 127.0.0.1 / 8000
    VIENEU_API_KEY                      đặt để bắt buộc ``Authorization: Bearer <key>`` hoặc ``X-API-Key``
    VIENEU_DEFAULT_VOICE                giọng dùng khi request không chỉ định ``voice``
    VIENEU_THREADS                      số luồng ONNX (đặt = số vCPU trên VPS)
    VIENEU_API_MAX_CHARS                giới hạn độ dài text (mặc định 3000)
    VIENEU_API_MAX_QUEUE                số request được xếp hàng chờ engine (mặc định 16; quá → 429)
    VIENEU_API_CORS_ORIGINS             danh sách origin (phẩy) được gọi từ trình duyệt; trống = tắt CORS
    VIENEU_USER_VOICES                  file JSON giọng người dùng (SDK cũng đọc file này)

Model được nạp MỘT lần lúc khởi động và dùng chung cho mọi request. Engine chỉ sinh
một request tại một thời điểm; các request khác xếp hàng trên event loop (asyncio.Lock),
không chiếm luồng của threadpool, nên /health và /voices vẫn trả lời khi engine bận.
"""
import argparse
import asyncio
import hmac
import io
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator, Literal, Optional
from urllib.parse import quote

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("Vieneu.API")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


SAMPLE_RATE = 48_000
MODEL_ID = "vieneu-v3-turbo"
MAX_TEXT_CHARS = max(1, _env_int("VIENEU_API_MAX_CHARS", 3000))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REF_SECONDS = 60.0
ALLOWED_REF_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
MIN_OUT_SR, MAX_OUT_SR = 8_000, 48_000

AudioFormat = Literal["wav", "pcm", "flac", "mp3", "opus", "aac"]
# fmt → (media type, cần ffmpeg?)
FORMATS = {
    "wav": ("audio/wav", False),
    "pcm": ("audio/pcm", False),      # raw PCM16 little-endian mono (giống OpenAI `pcm`)
    "flac": ("audio/flac", False),
    "mp3": ("audio/mpeg", True),
    "opus": ("audio/ogg", True),
    "aac": ("audio/aac", True),
}

# Engine CPU/ONNX chỉ sinh một request tại một thời điểm. asyncio.Lock xếp hàng các
# request ngay trên event loop (coroutine chờ, không chiếm luồng threadpool), nên
# một stream đang chạy không thể làm cạn threadpool hay treo server.
_infer_lock = asyncio.Lock()
_pending = 0   # số request đang chờ/đang chạy engine (để trả 429 khi quá tải)


# ── Model lifecycle ─────────────────────────────────────────────────────────────
def load_tts(threads: int = 0):
    """Nạp VieNeu v3 Turbo (ONNX/CPU khi không có torch, PyTorch khi có GPU)."""
    from vieneu import Vieneu
    kwargs = {}
    if threads and threads > 0:
        kwargs["threads"] = threads
    return Vieneu(**kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "tts", None) is None:
        t0 = time.time()
        print("⏳ Đang nạp VieNeu-TTS v3 Turbo...", flush=True)
        app.state.tts = await run_in_threadpool(load_tts, app.state.threads)
        tts = app.state.tts
        voices = tts.list_voices()
        n_user = sum(1 for v in voices if v["custom"])
        print(f"✅ Sẵn sàng sau {time.time() - t0:.1f}s | backend={tts.backend} | "
              f"{len(voices)} giọng ({n_user} giọng người dùng từ {tts.user_voices_path})", flush=True)
        dv = app.state.default_voice
        if dv and dv not in {v["id"] for v in voices}:
            print(f"⚠️  VIENEU_DEFAULT_VOICE='{dv}' không có trong danh sách giọng; request không chỉ định "
                  f"voice sẽ bị 404. Giọng có: {[v['id'] for v in voices]}", flush=True)
        if any(f for f, (_, needs) in FORMATS.items() if needs) and not shutil.which("ffmpeg"):
            print("ℹ️  Không thấy ffmpeg trên PATH: định dạng mp3/opus/aac sẽ trả 415; wav/pcm/flac vẫn dùng được.",
                  flush=True)
    yield


app = FastAPI(
    title="VieNeu-TTS API",
    description="TTS tiếng Việt (v3 Turbo, 48 kHz) với giọng có sẵn và giọng clone đã lưu.",
    version="1.1.0",
    lifespan=lifespan,
)
app.state.tts = None
app.state.threads = _env_int("VIENEU_THREADS", 0)
app.state.api_key = os.getenv("VIENEU_API_KEY", "").strip()
app.state.default_voice = os.getenv("VIENEU_DEFAULT_VOICE", "").strip()
app.state.max_queue = max(0, _env_int("VIENEU_API_MAX_QUEUE", 16))

_cors = [o.strip() for o in os.getenv("VIENEU_API_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Voice", "X-Sample-Rate", "X-Audio-Seconds", "X-Generation-Seconds"])


# ── OpenAI-style error bodies cho /v1/* ─────────────────────────────────────────
def _openai_error(status: int, message, code: Optional[str] = None) -> JSONResponse:
    if isinstance(message, dict):
        message = message.get("error") or str(message)
    return JSONResponse(status_code=status, content={"error": {
        "message": str(message), "type": "invalid_request_error" if status < 500 else "server_error",
        "param": None, "code": code}})


@app.exception_handler(StarletteHTTPException)
async def _http_exc(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/v1/"):
        return _openai_error(exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_exc(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/"):
        errs = exc.errors()
        msg = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', []) if p != 'body')}: {e.get('msg')}" for e in errs)
        return _openai_error(422, msg or "invalid request")
    return await request_validation_exception_handler(request, exc)


# ── Dependencies ────────────────────────────────────────────────────────────────
def get_tts(request: Request):
    tts = request.app.state.tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model chưa nạp xong, thử lại sau.")
    return tts


async def require_api_key(request: Request) -> None:
    key = request.app.state.api_key
    if not key:
        return
    auth = request.headers.get("authorization", "")
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not given or not hmac.compare_digest(given.encode("utf-8"), key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai API key (Authorization: Bearer <key> / X-API-Key).")


# ── Schemas ─────────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, description="Văn bản cần đọc")
    voice: Optional[str] = Field(None, description="ID giọng (xem GET /voices). Bỏ trống = giọng mặc định")
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int = Field(25, ge=1, le=200)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    max_chars: int = Field(256, ge=32, le=1024, description="Ngưỡng cắt chunk theo ký tự")
    sample_rate: Optional[int] = Field(None, ge=MIN_OUT_SR, le=MAX_OUT_SR,
                                       description="Resample đầu ra (vd 24000, 16000). Mặc định 48000")
    format: AudioFormat = Field("wav", description="wav | pcm (raw s16le mono) | flac | mp3 | opus | aac (3 loại cuối cần ffmpeg)")
    watermark: bool = Field(True, description="Đóng watermark ẩn (nếu thư viện có)")


class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: Optional[str] = None
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int = Field(25, ge=1, le=200)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    max_chars: int = Field(256, ge=32, le=1024)


class OpenAISpeechRequest(BaseModel):
    """Schema của OpenAI ``POST /v1/audio/speech`` (``model``/``speed`` được chấp nhận nhưng bỏ qua)."""
    model: str = MODEL_ID
    input: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: Optional[str] = None
    response_format: AudioFormat = "wav"
    speed: float = 1.0


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _resolve_voice(request: Request, tts, voice: Optional[str]) -> str:
    ids = [v["id"] for v in tts.list_voices()]
    name = (voice or "").strip() or request.app.state.default_voice or (getattr(tts, "_default_voice", None) or "")
    if name not in ids:
        raise HTTPException(status_code=404, detail={"error": f"Giọng '{name}' không tồn tại.", "available": ids})
    return name


def _clean_text(text: str, max_chars: int = 256) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Văn bản trống.")
    try:
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
        chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            raise HTTPException(status_code=422, detail="Văn bản không có nội dung đọc được (chỉ emoji/ký hiệu?).")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — normalizer lỗi thì để engine tự xử lý
        pass
    return text


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _resample(audio: np.ndarray, sr_out: Optional[int]) -> tuple:
    if not sr_out or sr_out == SAMPLE_RATE:
        return audio, SAMPLE_RATE
    import soxr
    return soxr.resample(np.asarray(audio, dtype=np.float32), SAMPLE_RATE, sr_out, quality="HQ").astype(np.float32), sr_out


def _sf_bytes(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format=fmt, subtype="PCM_16")
    return buf.getvalue()


def _ffmpeg_encode(wav_bytes: bytes, fmt: str) -> bytes:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise HTTPException(status_code=415, detail=f"Định dạng '{fmt}' cần ffmpeg trên server (không tìm thấy). "
                                                    f"Dùng wav, pcm hoặc flac.")
    args = {"mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k"],
            "opus": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "64k"],
            "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"]}[fmt]
    p = subprocess.run([exe, "-v", "error", "-f", "wav", "-i", "pipe:0", *args, "pipe:1"],
                       input=wav_bytes, capture_output=True, timeout=120)
    if p.returncode != 0 or not p.stdout:
        logger.error("ffmpeg failed: %s", p.stderr.decode("utf-8", "replace")[:500])
        raise HTTPException(status_code=500, detail=f"Mã hoá {fmt} thất bại trên server.")
    return p.stdout


def _encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return _pcm16_bytes(audio)
    if fmt == "wav":
        return _sf_bytes(audio, sr, "WAV")
    if fmt == "flac":
        return _sf_bytes(audio, sr, "FLAC")
    return _ffmpeg_encode(_sf_bytes(audio, sr, "WAV"), fmt)


def _stream_wav_header(sr: int) -> bytes:
    """Header WAV 44 byte với độ dài "chưa biết" (0xFFFFFFFF) để player phát liền khi data tới.

    Không dùng module ``wave``: khi đóng writer nó vá lại header thành 0 frame.
    """
    data_size = 0xFFFFFFFF - 36
    return (b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", data_size))


def _synthesize(tts, text: str, voice: str, req) -> tuple:
    """Chạy trong threadpool (caller đã giữ _infer_lock)."""
    t0 = time.perf_counter()
    audio = tts.infer(
        text, voice=voice,
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        repetition_penalty=req.repetition_penalty, max_chars=req.max_chars,
        apply_watermark=getattr(req, "watermark", True),
    )
    return np.asarray(audio, dtype=np.float32), time.perf_counter() - t0


class _EngineSlot:
    """Xếp hàng vào engine: 429 nếu hàng chờ đã đầy, còn lại chờ asyncio.Lock."""

    def __init__(self, request: Request):
        self.max_queue = request.app.state.max_queue

    async def __aenter__(self):
        global _pending
        if _pending >= self.max_queue:
            raise HTTPException(status_code=429, detail="Server đang bận (hàng chờ đầy), thử lại sau vài giây.",
                                headers={"Retry-After": "2"})
        _pending += 1
        try:
            await _infer_lock.acquire()
        except BaseException:
            _pending -= 1
            raise
        return self

    async def __aexit__(self, *exc):
        global _pending
        _pending -= 1
        _infer_lock.release()


async def _run_tts(request: Request, tts, text: str, voice: str, req) -> tuple:
    async with _EngineSlot(request):
        try:
            return await run_in_threadpool(_synthesize, tts, text, voice, req)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail="Lỗi sinh giọng trên server (xem log server).")


async def _audio_response(audio: np.ndarray, gen_s: float, voice: str, fmt: str, sample_rate: Optional[int]) -> Response:
    if audio.size == 0:
        raise HTTPException(status_code=422, detail="Không sinh được audio cho văn bản này.")
    audio, sr = _resample(audio, sample_rate)
    media_type, _ = FORMATS[fmt]
    body = await run_in_threadpool(_encode, audio, sr, fmt)
    headers = {
        "X-Voice": quote(voice, safe=""),        # percent-encoded: header chỉ chấp nhận latin-1
        "X-Sample-Rate": str(sr),
        "X-Audio-Seconds": f"{len(audio) / sr:.3f}",
        "X-Generation-Seconds": f"{gen_s:.3f}",
    }
    if fmt == "pcm":
        headers["X-PCM-Format"] = "s16le mono"
    else:
        ext = "ogg" if fmt == "opus" else fmt
        headers["Content-Disposition"] = f'inline; filename="speech.{ext}"'
    return Response(content=body, media_type=media_type, headers=headers)


# ── Endpoints ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"name": "VieNeu-TTS API", "model": MODEL_ID, "docs": "/docs",
            "endpoints": ["/health", "/voices", "/tts", "/tts/stream", "/v1/audio/speech", "/v1/models"]}


@app.get("/health")
async def health(request: Request):
    tts = request.app.state.tts
    return {
        "status": "ok" if tts is not None else "loading",
        "backend": getattr(tts, "backend", None),
        "sample_rate": SAMPLE_RATE,
        "voices": len(tts.list_voices()) if tts is not None else 0,
        "auth": bool(request.app.state.api_key),
        "busy": _infer_lock.locked(),
        "queue": _pending,
        "formats": [f for f, (_, needs) in FORMATS.items() if not needs or shutil.which("ffmpeg")],
    }


@app.get("/voices", dependencies=[Depends(require_api_key)])
async def list_voices(request: Request, tts=Depends(get_tts)):
    default = request.app.state.default_voice or getattr(tts, "_default_voice", None)
    return {"default": default, "voices": tts.list_voices()}


@app.post("/voices", status_code=201, dependencies=[Depends(require_api_key)])
async def add_voice(
    request: Request,
    name: str = Form(..., description="ID giọng mới, vd clone_A"),
    file: UploadFile = File(..., description="Audio mẫu 3-8 giây (.wav/.mp3/.flac/.ogg), một người nói"),
    description: str = Form("", description="Mô tả hiển thị, vd 'Nam · Giọng từ video X'"),
    gender: str = Form(""),
    denoise: bool = Form(True),
    tts=Depends(get_tts),
):
    name = (name or "").strip()
    if not name or len(name) > 64 or any(c in name for c in '/\\:*?"<>|\n\r\t'):
        raise HTTPException(status_code=422, detail="Tên giọng phải 1-64 ký tự, không chứa / \\ : * ? \" < > |")
    suffix = Path(file.filename or "").suffix.lower() or ".wav"
    if suffix not in ALLOWED_REF_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Định dạng {suffix} không hỗ trợ. Dùng {sorted(ALLOWED_REF_SUFFIXES)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail="File audio trống.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File quá lớn (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    tmp_dir = Path.home() / ".cache" / "vieneu" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="api_ref_", suffix=suffix, dir=str(tmp_dir))
    os.close(fd)
    try:
        Path(tmp_path).write_bytes(data)
        try:
            info = sf.info(tmp_path)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=422, detail="Không đọc được audio (file hỏng hoặc không phải wav/mp3/flac/ogg).")
        if info.duration < 1.0:
            raise HTTPException(status_code=422, detail=f"Audio mẫu quá ngắn ({info.duration:.1f}s); cần 3-8 giây.")
        if info.duration > MAX_REF_SECONDS:
            raise HTTPException(status_code=422, detail=f"Audio mẫu quá dài ({info.duration:.0f}s); hãy cắt đoạn 3-8 giây.")

        async with _EngineSlot(request):
            try:
                await run_in_threadpool(tts.add_voice, name, tmp_path, denoise=denoise,
                                        description=description, gender=gender, save=True)
            except ValueError as e:
                raise HTTPException(status_code=409, detail=str(e))
            except Exception:  # noqa: BLE001
                logger.exception("add_voice failed")
                raise HTTPException(status_code=500, detail="Không đăng ký được giọng (xem log server).")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:  # noqa: BLE001
            pass
    voice = next(v for v in tts.list_voices() if v["id"] == name)
    return {**voice, "ref_seconds": round(info.duration, 2), "persisted": True}


@app.delete("/voices/{name}", dependencies=[Depends(require_api_key)])
async def delete_voice(name: str, request: Request, tts=Depends(get_tts)):
    ids = {v["id"] for v in tts.list_voices()}
    if name not in ids:
        raise HTTPException(status_code=404, detail=f"Giọng '{name}' không tồn tại.")
    if not tts.is_user_voice(name):
        raise HTTPException(status_code=403, detail=f"'{name}' là giọng có sẵn, không xoá được.")
    async with _EngineSlot(request):
        await run_in_threadpool(tts.remove_voice, name, True)
    return {"removed": name}


@app.post("/tts", dependencies=[Depends(require_api_key)],
          responses={200: {"content": {m: {} for m, _ in FORMATS.values()}, "description": "Audio"}})
async def tts_endpoint(req: TTSRequest, request: Request, tts=Depends(get_tts)):
    text = _clean_text(req.text, req.max_chars)
    voice = _resolve_voice(request, tts, req.voice)
    audio, gen_s = await _run_tts(request, tts, text, voice, req)
    return await _audio_response(audio, gen_s, voice, req.format, req.sample_rate)


@app.post("/tts/stream", dependencies=[Depends(require_api_key)],
          responses={200: {"content": {"audio/wav": {}}, "description": "WAV stream 48 kHz"}})
async def tts_stream(req: StreamRequest, request: Request, tts=Depends(get_tts)):
    text = _clean_text(req.text, req.max_chars)
    voice = _resolve_voice(request, tts, req.voice)

    def gen() -> Iterator[bytes]:
        for chunk in tts.infer_stream(
            text, voice=voice,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty, max_chars=req.max_chars,
        ):
            if chunk is None or len(chunk) == 0:
                continue
            yield _pcm16_bytes(chunk)

    slot = _EngineSlot(request)
    await slot.__aenter__()          # 429 nếu quá tải; chờ engine rảnh trước khi gửi header

    async def agen() -> AsyncIterator[bytes]:
        try:
            yield _stream_wav_header(SAMPLE_RATE)
            async for chunk in iterate_in_threadpool(gen()):
                yield chunk
        except Exception:  # noqa: BLE001
            logger.exception("stream failed")
            raise
        finally:
            await slot.__aexit__(None, None, None)   # kể cả khi client ngắt kết nối

    return StreamingResponse(agen(), media_type="audio/wav",
                             headers={"X-Voice": quote(voice, safe=""), "X-Sample-Rate": str(SAMPLE_RATE)})


@app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)],
          responses={200: {"content": {m: {} for m, _ in FORMATS.values()}, "description": "Audio"}})
async def openai_speech(req: OpenAISpeechRequest, request: Request, tts=Depends(get_tts)):
    """Tương thích OpenAI: ``{"model": "...", "input": "...", "voice": "clone_A", "response_format": "wav"}``."""
    inner = TTSRequest(text=req.input, voice=req.voice, format=req.response_format)
    text = _clean_text(inner.text, inner.max_chars)
    voice = _resolve_voice(request, tts, inner.voice)
    audio, gen_s = await _run_tts(request, tts, text, voice, inner)
    return await _audio_response(audio, gen_s, voice, inner.format, None)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def openai_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "vieneu"}]}


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="VieNeu-TTS HTTP API server")
    p.add_argument("--host", default=os.getenv("VIENEU_API_HOST", "127.0.0.1"),
                   help="0.0.0.0 để mở ra mạng (mặc định 127.0.0.1)")
    p.add_argument("--port", type=int, default=_env_int("VIENEU_API_PORT", 8000))
    p.add_argument("--threads", type=int, default=_env_int("VIENEU_THREADS", 0),
                   help="Số luồng ONNX; trên VPS đặt = số vCPU (mặc định engine = cpu_count//2, tối đa 8)")
    p.add_argument("--api-key", default=os.getenv("VIENEU_API_KEY", ""),
                   help="Bắt buộc Authorization: Bearer <key> (mặc định: tắt)")
    p.add_argument("--default-voice", default=os.getenv("VIENEU_DEFAULT_VOICE", ""),
                   help="Giọng dùng khi request không chỉ định voice")
    p.add_argument("--max-queue", type=int, default=_env_int("VIENEU_API_MAX_QUEUE", 16),
                   help="Số request được xếp hàng chờ engine; vượt quá trả 429")
    args = p.parse_args(argv)

    app.state.threads = args.threads
    app.state.api_key = (args.api_key or "").strip()
    app.state.default_voice = (args.default_voice or "").strip()
    app.state.max_queue = max(0, args.max_queue)
    if args.host not in ("127.0.0.1", "localhost") and not app.state.api_key:
        print("⚠️  Server mở ra mạng mà không có API key. Đặt --api-key hoặc VIENEU_API_KEY nếu VPS có IP public.",
              flush=True)
    print(f"🌍 VieNeu-TTS API: http://{args.host}:{args.port}  (docs: /docs)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
