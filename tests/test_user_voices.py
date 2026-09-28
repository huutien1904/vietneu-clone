"""Giọng người dùng (clone đã lưu) cho v3 Turbo: nạp gộp, không ghi đè giọng gốc, lưu atomic."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vieneu.v3turbo import V3TurboVieNeuTTS, default_user_voices_path


def _bare(user_path: Path) -> V3TurboVieNeuTTS:
    """Instance không nạp engine: chỉ phần quản lý giọng."""
    t = object.__new__(V3TurboVieNeuTTS)
    t.default_style = "tu_nhien"
    t._preset_voices = {}
    t._default_voice = None
    t._user_voice_names = set()
    t._builtin_voice_names = set()
    t._removed_user_voices = set()
    t.user_voices_path = user_path
    return t


class _FakeEngine:
    def __init__(self, seed=0.5):
        self.seed = seed
        self.calls = []

    def prepare_reference(self, path, denoise=True, use_ref_codes=True):
        self.calls.append((path, denoise, use_ref_codes))
        return np.full(192, self.seed, np.float32), np.array([[1, 2, 3, 4]] * 5, np.int64)


def _voice(seed: float) -> dict:
    return {"description": "Nam · test", "gender": "male", "style": "tu_nhien",
            "speaker_emb": [seed] * 192, "codes": [[1, 2, 3, 4]] * 3}


def _ref_wav(tmp_path: Path, name="ref.wav") -> Path:
    p = tmp_path / name
    sr = 16000
    wav = np.zeros(sr * 2, np.float32)
    wav[sr // 2: sr * 3 // 2] = 0.3
    sf.write(p, wav, sr)
    return p


def test_default_path_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("VIENEU_USER_VOICES", raising=False)
    assert default_user_voices_path() == Path.home() / ".cache" / "vieneu" / "user_voices.json"
    monkeypatch.setenv("VIENEU_USER_VOICES", str(tmp_path / "v.json"))
    assert default_user_voices_path() == tmp_path / "v.json"


def test_builtin_voices_load_and_default(tmp_path):
    t = _bare(tmp_path / "none.json")
    t._load_v3_voices()
    ids = [i for _, i in t.list_preset_voices()]
    assert len(ids) >= 20 and "Adam" in ids
    assert t._default_voice in ids
    assert not any(v["custom"] for v in t.list_voices())
    assert t._builtin_voice_names == set(ids)


def test_user_voices_merge_and_collision(tmp_path):
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {
        "clone_A": _voice(0.5),
        "Adam": _voice(0.9),            # trùng giọng có sẵn → bỏ qua
        "no_emb": {"description": "x", "codes": [[1]]},   # thiếu speaker_emb → bỏ qua
    }}, ensure_ascii=False), encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()
    by_id = {v["id"]: v for v in t.list_voices()}
    assert by_id["clone_A"]["custom"] is True
    assert by_id["clone_A"]["label"] == "clone_A — Nam · test"
    assert t.is_user_voice("clone_A") and not t.is_user_voice("Adam")
    assert "no_emb" not in by_id
    # giọng gốc Adam không bị ghi đè bởi file người dùng
    assert float(t.get_preset_voice("Adam")["speaker_emb"][0]) != pytest.approx(0.9)
    assert t.get_preset_voice("clone_A")["speaker_emb"].dtype == np.float32
    assert t.get_preset_voice("clone_A")["codes"].shape == (3, 4)


@pytest.mark.parametrize("content", [
    "{not json",
    "[]",
    "null",
    '"str"',
    '{"presets": []}',
    '{"presets": {"x": null}}',
    '{"presets": {"x": "oops"}}',
    '{"presets": {"x": {"speaker_emb": "abc"}}}',
    '{"presets": {"x": {"speaker_emb": [[1, 2], [3]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [[1, 2, 3], [1, 2]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [["a", "b"]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [1, 2, 3]}}}',
    '{"presets": {"x": {"speaker_emb": []}}}',
])
def test_malformed_user_file_never_raises(tmp_path, content):
    up = tmp_path / "user_voices.json"
    up.write_text(content, encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()          # không ném lỗi
    assert "x" not in t._preset_voices
    assert len(t._preset_voices) >= 20 and t._default_voice in t._preset_voices


def test_malformed_entry_does_not_block_healthy_sibling(tmp_path):
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {"bad": {"speaker_emb": "abc"}, "clone_A": _voice(0.4)}}), encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()
    assert t.is_user_voice("clone_A") and "bad" not in t._preset_voices


def test_add_voice_happy_path_and_replace(tmp_path):
    up = tmp_path / "user_voices.json"
    t = _bare(up)
    t._load_v3_voices()
    t.engine = _FakeEngine(0.5)
    ref = _ref_wav(tmp_path)

    assert t.add_voice(" clone_A ", ref, description="d1", save=True) == "clone_A"
    assert t.is_user_voice("clone_A")
    assert t.get_preset_voice("clone_A")["speaker_emb"].shape == (192,)
    assert t.get_preset_voice("clone_A")["codes"].shape == (5, 4)
    assert t._default_voice != "clone_A"          # đã có giọng mặc định gốc
    assert json.loads(up.read_text(encoding="utf-8"))["presets"]["clone_A"]["description"] == "d1"
    assert t.engine.calls[0][1] is True           # denoise mặc định
    assert not list(tmp_path.glob("**/temp_clone_optimized_*"))  # file tạm đã dọn

    # thay thế giọng người dùng cùng tên
    t.engine = _FakeEngine(0.7)
    t.add_voice("clone_A", ref, description="d2", denoise=False, save=True)
    assert float(t.get_preset_voice("clone_A")["speaker_emb"][0]) == pytest.approx(0.7)
    assert json.loads(up.read_text(encoding="utf-8"))["presets"]["clone_A"]["description"] == "d2"
    assert t.engine.calls[0][1] is False

    # tên trùng giọng có sẵn → lỗi, kể cả khi đã remove khỏi phiên
    with pytest.raises(ValueError):
        t.add_voice("Adam", ref)
    t.remove_voice("Adam")
    with pytest.raises(ValueError):
        t.add_voice("Adam", ref)
    with pytest.raises(ValueError):
        t.add_voice("   ", ref)


def test_add_voice_save_failure_rolls_back(tmp_path):
    up = tmp_path / "dir_as_file"
    up.mkdir()                                     # ghi file vào đường dẫn là thư mục → thất bại
    t = _bare(up)
    t._load_v3_voices()
    t.engine = _FakeEngine()
    with pytest.raises(Exception):
        t.add_voice("clone_X", _ref_wav(tmp_path), save=True)
    assert "clone_X" not in t._preset_voices and not t.is_user_voice("clone_X")


def test_save_user_voices_merges_with_disk(tmp_path):
    up = tmp_path / "sub" / "user_voices.json"
    t = _bare(up)
    t._load_v3_voices()
    t._preset_voices["clone_B"] = {"description": "d", "gender": "", "style": "tu_nhien",
                                   "speaker_emb": np.full(192, 0.25, np.float32),
                                   "codes": np.array([[7, 8]], np.int64)}
    t._user_voice_names.add("clone_B")
    out = t.save_user_voices()
    assert Path(out) == up and up.exists()
    data = json.loads(up.read_text(encoding="utf-8"))
    assert list(data["presets"]) == ["clone_B"]
    assert data["presets"]["clone_B"]["speaker_emb"][0] == pytest.approx(0.25)
    assert not list(up.parent.glob(".user_voices_*"))   # file tạm đã dọn

    # tiến trình khác thêm clone_C vào file → save của phiên này phải giữ lại clone_C
    data["presets"]["clone_C"] = _voice(0.6)
    up.write_text(json.dumps(data), encoding="utf-8")
    t.save_user_voices()
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_B", "clone_C"}

    # nạp lại ở phiên mới → cả hai là custom; remove(save=True) chỉ bỏ giọng đã xoá
    t2 = _bare(up)
    t2._load_v3_voices()
    assert t2.is_user_voice("clone_B") and t2.is_user_voice("clone_C")
    t2.remove_voice("clone_B", save=True)
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_C"}
    assert "Adam" in t2._preset_voices

    # giọng có sẵn lỡ có trong file người dùng thì không được ghi lại
    up.write_text(json.dumps({"presets": {"Adam": _voice(0.9), "clone_C": _voice(0.6)}}), encoding="utf-8")
    t2.save_user_voices()
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_C"}


def test_save_voices_exports_everything_to_explicit_path(tmp_path):
    t = _bare(tmp_path / "u.json")
    t._load_v3_voices()
    out = tmp_path / "all.json"
    t.save_voices(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["presets"]) >= 20 and data["default_voice"] == t._default_voice
