"""API server (apps/api_server.py) với engine giả — không cần model, chạy torch-free."""
import io
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from apps import api_server


class FakeTTS:
    sample_rate = 48_000
    backend = "fake"

    def __init__(self):
        self._default_voice = "Adam"
        self.user_voices_path = Path("fake_user_voices.json")
        self._voices = {"Adam": {"description": "Nam · Nam", "gender": "male", "custom": False},
                        "Minh Đức": {"description": "Nam · Bắc", "gender": "male", "custom": False},
                        "clone_A": {"description": "Giọng clone", "gender": "", "custom": True}}
        self.calls = []
        self.saved = []

    def list_voices(self):
        return [{"id": n, "label": f"{n} — {v['description']}", "description": v["description"],
                 "gender": v["gender"], "custom": v["custom"]} for n, v in self._voices.items()]

    def list_preset_voices(self):
        return [(v["label"], v["id"]) for v in self.list_voices()]

    def is_user_voice(self, name):
        return self._voices.get(name, {}).get("custom", False)

    def infer(self, text, voice=None, **kw):
        self.calls.append(("infer", text, voice, kw))
        t = np.arange(4800) / 48_000.0          # 0.1 s
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def infer_stream(self, text, voice=None, **kw):
        self.calls.append(("stream", text, voice, kw))
        for _ in range(3):
            yield np.full(480, 0.25, np.float32)

    def add_voice(self, name, ref_audio, **kw):
        assert Path(ref_audio).exists()
        if name in self._voices and not self._voices[name]["custom"]:
            raise ValueError("built-in")
        self._voices[name] = {"description": kw.get("description", ""), "gender": kw.get("gender", ""), "custom": True}
        self.saved.append((name, kw.get("save")))
        return name

    def remove_voice(self, name, save=False):
        self._voices.pop(name, None)
        self.saved.append((name, "removed", save))


@pytest.fixture
def client():
    fake = FakeTTS()
    api_server.app.state.tts = fake
    api_server.app.state.api_key = ""
    api_server.app.state.default_voice = ""
    api_server.app.state.max_queue = 16
    c = TestClient(api_server.app)
    c.fake = fake
    yield c
    api_server.app.state.tts = None


def _wav_bytes(seconds=2.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), np.float32) + 0.1, sr, format="WAV")
    return buf.getvalue()


def test_health_and_voices(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["voices"] == 3 and h["auth"] is False and h["busy"] is False
    assert "wav" in h["formats"] and "pcm" in h["formats"] and "flac" in h["formats"]
    v = client.get("/voices").json()
    assert v["default"] == "Adam"
    assert {x["id"] for x in v["voices"]} == {"Adam", "Minh Đức", "clone_A"}
    assert next(x for x in v["voices"] if x["id"] == "clone_A")["custom"] is True


def test_tts_wav_with_custom_voice(client):
    r = client.post("/tts", json={"text": "Xin chào", "voice": "clone_A"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-voice"] == "clone_A"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 48_000 and abs(len(data) / sr - 0.1) < 1e-3
    assert client.fake.calls[-1][2] == "clone_A"
    assert client.fake.calls[-1][3]["apply_watermark"] is True


def test_vietnamese_voice_name_in_headers(client):
    r = client.post("/tts", json={"text": "Xin chào", "voice": "Minh Đức"})
    assert r.status_code == 200, r.text
    assert r.headers["x-voice"] == "Minh%20%C4%90%E1%BB%A9c"
    with client.stream("POST", "/tts/stream", json={"text": "Xin chào", "voice": "Minh Đức"}) as s:
        assert s.status_code == 200 and s.headers["x-voice"] == "Minh%20%C4%90%E1%BB%A9c"
        b"".join(s.iter_bytes())


def test_tts_default_voice_resample_and_formats(client):
    r = client.post("/tts", json={"text": "A", "sample_rate": 16000})
    assert r.status_code == 200
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 16000 and abs(len(data) / sr - 0.1) < 2e-3
    assert client.fake.calls[-1][2] == "Adam"

    r = client.post("/tts", json={"text": "A", "format": "pcm"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-pcm-format"] == "s16le mono" and len(r.content) == 4800 * 2

    r = client.post("/tts", json={"text": "A", "format": "flac"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/flac")
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 48_000 and len(data) == 4800


def test_tts_mp3_requires_ffmpeg(client, monkeypatch):
    monkeypatch.setattr(api_server.shutil, "which", lambda name: None)
    r = client.post("/tts", json={"text": "A", "format": "mp3"})
    assert r.status_code == 415
    assert "mp3" not in client.get("/health").json()["formats"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg không có trên máy này")
def test_tts_mp3_with_ffmpeg(client):
    r = client.post("/tts", json={"text": "A", "format": "mp3"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/mpeg") and len(r.content) > 100
    assert r.content[:3] == b"ID3" or r.content[0] == 0xFF


def test_tts_validation(client):
    assert client.post("/tts", json={"text": "   "}).status_code == 422
    assert client.post("/tts", json={"text": ""}).status_code == 422
    assert client.post("/tts", json={"text": "😀🎉"}).status_code == 422       # không có gì để đọc
    r = client.post("/tts", json={"text": "hi", "voice": "khong_co"})
    assert r.status_code == 404 and "available" in r.json()["detail"]
    assert client.post("/tts", json={"text": "x" * (api_server.MAX_TEXT_CHARS + 1)}).status_code == 422
    assert client.post("/tts", json={"text": "hi", "sample_rate": 100}).status_code == 422
    assert client.post("/tts", json={"text": "hi", "format": "wma"}).status_code == 422


def test_openai_compatible(client):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Xin chào", "voice": "clone_A"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    r = client.post("/v1/audio/speech", json={"input": "Xin chào", "voice": "clone_A", "response_format": "pcm"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/pcm")
    # lỗi theo schema OpenAI {"error": {...}}
    r = client.post("/v1/audio/speech", json={"input": "Xin chào", "voice": "alloy"})
    assert r.status_code == 404 and "message" in r.json()["error"] and r.json()["error"]["type"] == "invalid_request_error"
    r = client.post("/v1/audio/speech", json={"input": "Xin chào", "response_format": "wma"})
    assert r.status_code == 422 and "response_format" in r.json()["error"]["message"]
    m = client.get("/v1/models").json()
    assert m["data"][0]["id"] == api_server.MODEL_ID
    # giọng mặc định cấu hình qua env/--default-voice
    api_server.app.state.default_voice = "clone_A"
    r = client.post("/v1/audio/speech", json={"input": "Xin chào"})
    assert r.status_code == 200 and r.headers["x-voice"] == "clone_A"


def test_stream_returns_wav_header_then_pcm(client):
    with client.stream("POST", "/tts/stream", json={"text": "Xin chào", "voice": "clone_A"}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    riff_size, = struct.unpack("<I", body[4:8]); data_size, = struct.unpack("<I", body[40:44])
    assert data_size == 0xFFFFFFFF - 36 and riff_size == 0xFFFFFFFF   # độ dài "chưa biết", không phải 0
    assert struct.unpack("<I", body[24:28])[0] == 48_000
    assert len(body) == 44 + 3 * 480 * 2
    assert not api_server._infer_lock.locked() and api_server._pending == 0   # lock đã nhả sau stream


def test_queue_full_returns_429(client):
    api_server.app.state.max_queue = 0
    r = client.post("/tts", json={"text": "hi"})
    assert r.status_code == 429 and r.headers.get("retry-after")
    with client.stream("POST", "/tts/stream", json={"text": "hi"}) as s:
        assert s.status_code == 429
    assert api_server._pending == 0 and not api_server._infer_lock.locked()


def test_api_key_required_on_all_protected_routes(client):
    api_server.app.state.api_key = "secret"
    assert client.get("/health").status_code == 200          # health luôn mở
    for method, url, kw in [
        ("get", "/voices", {}),
        ("post", "/tts", {"json": {"text": "hi"}}),
        ("post", "/tts/stream", {"json": {"text": "hi"}}),
        ("post", "/v1/audio/speech", {"json": {"input": "hi"}}),
        ("get", "/v1/models", {}),
        ("delete", "/voices/clone_A", {}),
        ("post", "/voices", {"data": {"name": "x"}, "files": {"file": ("r.wav", _wav_bytes(), "audio/wav")}}),
    ]:
        assert getattr(client, method)(url, **kw).status_code == 401, url
    assert client.get("/voices", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.get("/voices", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/voices", headers={"X-API-Key": "wrong"}).status_code == 401
    # header non-ASCII tới server dưới dạng latin-1 → phải là 401, không phải 500
    assert client.get("/voices", headers={"X-API-Key": "sécret".encode("latin-1")}).status_code == 401
    r = client.post("/v1/audio/speech", json={"input": "hi"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401 and "error" in r.json()


def test_add_and_delete_voice(client):
    r = client.post("/voices", data={"name": "clone_B", "description": "Nữ · test"},
                    files={"file": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "clone_B" and body["custom"] is True and body["persisted"] is True
    assert "saved_to" not in body                      # không lộ đường dẫn server
    assert client.fake.saved[-1] == ("clone_B", True)  # đã lưu bền
    assert "clone_B" in {v["id"] for v in client.get("/voices").json()["voices"]}

    # tên trùng giọng có sẵn → 409; tên xấu → 422; định dạng lạ → 415; audio quá ngắn/quá dài → 422
    r = client.post("/voices", data={"name": "Adam"}, files={"file": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 409
    r = client.post("/voices", data={"name": "../x"}, files={"file": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 422
    r = client.post("/voices", data={"name": "c"}, files={"file": ("ref.mp4", b"0000", "video/mp4")})
    assert r.status_code == 415
    r = client.post("/voices", data={"name": "c"}, files={"file": ("ref.wav", _wav_bytes(0.3), "audio/wav")})
    assert r.status_code == 422
    r = client.post("/voices", data={"name": "c"}, files={"file": ("ref.wav", _wav_bytes(70, 8000), "audio/wav")})
    assert r.status_code == 422
    r = client.post("/voices", data={"name": "c"}, files={"file": ("ref.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 422

    assert client.delete("/voices/Adam").status_code == 403
    assert client.delete("/voices/khong_co").status_code == 404
    assert client.delete("/voices/clone_B").status_code == 200
    assert client.fake.saved[-1] == ("clone_B", "removed", True)
    assert "clone_B" not in {v["id"] for v in client.get("/voices").json()["voices"]}
    assert api_server._pending == 0 and not api_server._infer_lock.locked()
