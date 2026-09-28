"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")   # clone a voice
    wav = tts.infer("Xin chào", voice="Xuân Vĩnh")     # preset voice
    tts.save(wav, "out.wav")

``style`` is DEPRECATED on v3 Turbo: the speaking style is already implied by the
reference voice (speaker embedding + reference codes), so generation is always the
natural style. The argument is still accepted everywhere for backward compatibility,
but it is ignored.
"""
import logging
import os
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from ._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence, max_expected_frames


def _cap_frames(sampling: dict, cap: int) -> dict:
    """Bản sao ``sampling`` với ``max_new_frames`` chặn trên bởi ``cap``.

    Chặn phần "nói thêm" khi stop token bắn trượt trên chunk ngắn (xem
    :func:`vieneu_utils.core_utils.max_expected_frames`); chunk dài không bị
    ảnh hưởng vì cap luôn vượt ``max_new_frames`` mặc định.
    """
    out = dict(sampling)
    out["max_new_frames"] = min(out.get("max_new_frames", 300), cap)
    return out

logger = logging.getLogger("Vieneu.V3Turbo")

# Giọng người dùng (clone rồi lưu bằng ``add_voice(..., save=True)``) nằm NGOÀI
# package asset: không ghi đè ``assets/voices_v3_turbo.json`` (20 giọng gốc) và
# sống sót qua ``pip install -U vieneu``. SDK, web UI và API server đều tự nạp
# file này nên giọng đã lưu xuất hiện ở mọi nơi.
USER_VOICES_ENV = "VIENEU_USER_VOICES"


def default_user_voices_path() -> Path:
    """File giọng người dùng: ``$VIENEU_USER_VOICES`` hoặc ``~/.cache/vieneu/user_voices.json``."""
    env = os.environ.get(USER_VOICES_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "vieneu" / "user_voices.json"


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)."""

    @staticmethod
    def _preclean_reference_audio(
        ref_audio: Union[str, Path],
        *,
        top_db: int = 30,
        out_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Trim silence at the edges and save a cleaned reference clip.
        """
        import os
        import tempfile
        import soundfile as sf

        src = Path(ref_audio)
        if not src.exists():
            return str(src)

        wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)

        try:
            import librosa
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        except Exception:
            nonzero = np.flatnonzero(np.abs(wav) > 1e-5)
            if nonzero.size == 0:
                wav_trimmed = wav
            else:
                start = max(int(nonzero[0]) - int(0.01 * sr), 0)
                end = min(int(nonzero[-1]) + int(0.01 * sr), len(wav))
                wav_trimmed = wav[start:end]

        if out_path is None:
            try:
                secure_dir = Path.home() / ".cache" / "vieneu" / "tmp"
            except Exception:
                secure_dir = Path("./outputs/tmp")

            secure_dir.mkdir(parents=True, exist_ok=True)
            try:
                secure_dir.chmod(0o700)
            except Exception:
                pass

            fd, out_path = tempfile.mkstemp(
                prefix="temp_clone_optimized_",
                suffix=".wav",
                dir=str(secure_dir)
            )
            os.close(fd)
            out_path = Path(out_path)

        sf.write(str(out_path), wav_trimmed, sr)
        return str(out_path)

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        model_subfolder: str = "update",
        moss_tokenizer: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "int8",   # ONNX/CPU backbone: "int8" (mặc định, nhanh ~3x/frame, nhỏ 4x) | "fp32" (chất-lượng-tối-đa)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        max_batch_size: int = 32,   # GPU/PyTorch: trần số chunk gộp vào một forward (static batching). Batch thực = min(số_chunk, max_batch_size). Bỏ qua trên CPU/ONNX.
        user_voices: Optional[Union[str, Path]] = None,   # file JSON giọng người dùng (clone đã lưu); None → default_user_voices_path()
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000

        # `precision` chỉ áp cho đường ONNX/CPU (chọn subfolder graph int8 vs fp32).
        # Đường PyTorch/GPU dùng torch fp32/bf16, không liên quan.
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_int8")

        if device in (None, "auto"):
            try:
                import torch
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev_type = "cpu"
        else:
            dev_type = "cuda" if "cuda" in str(device).lower() else str(device).lower()
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        # Style is deprecated on v3 Turbo: it is implied by the reference (speaker
        # embedding + ref codes), so every generation uses the natural style. Kept
        # only as voice metadata / for backward-compatible call signatures.
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self._user_voice_names: set = set()
        self._builtin_voice_names: set = set()
        self._removed_user_voices: set = set()
        self.user_voices_path: Path = (
            Path(user_voices).expanduser() if user_voices else default_user_voices_path()
        )
        self._load_v3_voices()

        # Static-batching (GPU/PyTorch). Dựng lười ở lần batch đầu; None trên CPU/ONNX.
        self.max_batch_size = max(1, int(max_batch_size))
        self._batch_engine = None

    # ── Preset voices (speaker embedding + reference codes) ─────────────────────
    @staticmethod
    def builtin_voices_path() -> Path:
        """File giọng có sẵn đi kèm package (không nên ghi vào)."""
        return Path(__file__).parent / "assets" / "voices_v3_turbo.json"

    def _load_voices_file(self, path: Path, *, user: bool) -> int:
        """Nạp một file giọng (schema ``{"default_voice"?, "presets": {name: {...}}}``).

        Mỗi giọng mang ``speaker_emb`` 192-d và ``codes`` tham chiếu đã mã hoá.
        ``user=True``: giọng người dùng — tên trùng giọng có sẵn bị bỏ qua (không
        cho ghi đè giọng gốc) và được đánh dấu để ``save_user_voices`` chỉ lưu chúng.
        Trả về số giọng đã nạp; file thiếu/hỏng → 0 và chỉ cảnh báo, không ném lỗi.
        """
        import json
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ Không đọc được file giọng {path}: {e}")
            return 0
        presets = data.get("presets") if isinstance(data, dict) else None
        if presets is None:
            presets = {}
        if not isinstance(presets, dict):
            logger.warning(f"⚠️ File giọng {path} sai schema (cần {{\"presets\": {{tên: {{...}}}}}}) — bỏ qua.")
            return 0
        n = 0
        for name, v in presets.items():
            if user and name in self._builtin_voice_names:
                logger.warning(f"⚠️ Giọng người dùng '{name}' trùng tên giọng có sẵn — bỏ qua.")
                continue
            if not isinstance(v, dict) or v.get("speaker_emb") is None:
                logger.warning(f"⚠️ Giọng '{name}' trong {path} sai schema hoặc thiếu speaker_emb — bỏ qua.")
                continue
            try:
                emb = np.asarray(v["speaker_emb"], dtype=np.float32).reshape(-1)
                codes = v.get("codes")
                codes = np.asarray(codes, dtype=np.int64) if codes is not None else None
                if emb.size == 0 or (codes is not None and codes.ndim != 2):
                    raise ValueError("speaker_emb rỗng hoặc codes không phải ma trận 2 chiều")
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(f"⚠️ Giọng '{name}' trong {path} có speaker_emb/codes không hợp lệ ({e}) — bỏ qua.")
                continue
            self._preset_voices[name] = {
                "description": v.get("description", "") or "",
                "gender": v.get("gender", "") or "",
                "style": v.get("style", self.default_style),
                "speaker_emb": emb,
                "codes": codes,
            }
            if user:
                self._user_voice_names.add(name)
            else:
                self._builtin_voice_names.add(name)
            n += 1
        if not user and isinstance(data, dict) and data.get("default_voice"):
            self._default_voice = data.get("default_voice")
        return n

    def _load_v3_voices(self) -> None:
        """Nạp giọng có sẵn (``assets/voices_v3_turbo.json``) rồi giọng người dùng (``user_voices_path``)."""
        n_builtin = self._load_voices_file(self.builtin_voices_path(), user=False)
        n_user = self._load_voices_file(self.user_voices_path, user=True)
        if self._default_voice not in self._preset_voices:
            self._default_voice = next(iter(self._preset_voices), None)
        logger.info(f"📢 Loaded {n_builtin} preset voices + {n_user} user voices "
                    f"(default: {self._default_voice}; user file: {self.user_voices_path})")

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for every voice (built-in + user)."""
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in list(self._preset_voices.items())]

    def list_voices(self) -> List[dict]:
        """Return ``[{"id", "label", "description", "gender", "custom"}, ...]`` (``custom`` = giọng người dùng)."""
        out = []
        for n, v in list(self._preset_voices.items()):
            desc = v.get("description", "") or ""
            out.append({
                "id": n,
                "label": f"{n} — {desc}" if desc else n,
                "description": desc,
                "gender": v.get("gender", "") or "",
                "custom": n in self._user_voice_names,
            })
        return out

    def is_user_voice(self, name: str) -> bool:
        """True nếu ``name`` là giọng người dùng (clone đã đăng ký), không phải giọng có sẵn."""
        return name in self._user_voice_names

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a wav → ``(speaker_emb, ref_codes)``.
        """
        import os
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            return self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=True)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass

    def denoise(self, ref_audio: Union[str, Path], out_path: Optional[Union[str, Path]] = None,
                max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Clean up a reference clip and return ``(wav, sample_rate)`` at 44.1 kHz.

        Pass ``out_path`` to also save the cleaned wav. ``max_seconds`` optionally
        trims the clip first. Use the result as a nicer reference for cloning.
        """
        import soundfile as sf
        den = getattr(self.engine, "denoiser", None)
        if den is None:
            raise RuntimeError("Denoiser không khả dụng trên backend này.")
        wav, sr = sf.read(str(ref_audio))
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if max_seconds and len(wav) > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]
        clean = den.denoise(wav, sr)          # -> float32 mono @ 44100
        if out_path is not None:
            sf.write(str(out_path), clean, 44100)
        return clean, 44100

    def add_voice(self, name: str, ref_audio: Union[str, Path], *, denoise: bool = True,
                  use_ref_codes: bool = True, description: str = "", gender: str = "",
                  style: Any = None, save: bool = False) -> str:
        """Register a custom voice under ``name`` for use as ``infer(..., voice=name)``.

        ``ref_audio`` is enrolled once (denoised + trimmed, then speaker embedding +
        reference codes are extracted). Pass ``denoise=False`` if the clip is already
        clean. ``save=True`` ghi giọng vào file giọng người dùng (``user_voices_path``:
        ``$VIENEU_USER_VOICES`` hoặc ``~/.cache/vieneu/user_voices.json``) để các phiên
        sau (SDK, web UI, API server) tự nạp lại. File asset gốc không bị đụng.

        ``name`` không được trùng giọng có sẵn (không cho ghi đè giọng gốc); trùng
        giọng người dùng cũ thì thay thế.

        ``style`` is deprecated: it is stored as metadata only and never changes how
        the voice is synthesized (the style is implied by the reference itself).
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("Tên giọng không được để trống.")
        if name in self._builtin_voice_names or (name in self._preset_voices and name not in self._user_voice_names):
            raise ValueError(f"'{name}' là giọng có sẵn, không thể ghi đè. Hãy chọn tên khác.")
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            speaker_emb, ref_codes = self.engine.prepare_reference(
                str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass
        previous = self._preset_voices.get(name)
        self._preset_voices[name] = {
            "description": description or "",
            "gender": gender or "",
            "style": style or self.default_style,   # metadata only (deprecated)
            "speaker_emb": np.asarray(speaker_emb, dtype=np.float32).reshape(-1),
            "codes": None if ref_codes is None else np.asarray(ref_codes, dtype=np.int64),
        }
        self._user_voice_names.add(name)
        self._removed_user_voices.discard(name)
        if not self._default_voice:
            self._default_voice = name
        if save:
            try:
                self.save_user_voices()
            except Exception:
                # Ghi đĩa thất bại → không giữ giọng "nửa vời" trong phiên; báo lỗi cho caller.
                if previous is None:
                    self._preset_voices.pop(name, None)
                    self._user_voice_names.discard(name)
                    if self._default_voice == name:
                        self._default_voice = next(iter(self._preset_voices), None)
                else:
                    self._preset_voices[name] = previous
                raise
        logger.info(f"➕ Added voice '{name}'.")
        return name

    def remove_voice(self, name: str, save: bool = False) -> None:
        """Remove a registered voice by name.

        Giọng có sẵn chỉ bị gỡ khỏi phiên hiện tại (file asset không đổi);
        ``save=True`` ghi lại file giọng người dùng (bỏ giọng vừa xoá nếu là giọng người dùng).
        """
        self._preset_voices.pop(name, None)
        if name in self._user_voice_names or name not in self._builtin_voice_names:
            self._removed_user_voices.add(name)
        self._user_voice_names.discard(name)
        if self._default_voice == name:
            self._default_voice = next(iter(self._preset_voices), None)
        if save:
            self.save_user_voices()

    def _voice_to_json(self, v: dict) -> dict:
        emb = v.get("speaker_emb")
        codes = v.get("codes")
        return {
            "description": v.get("description", "") or "",
            "gender": v.get("gender", "") or "",
            "style": v.get("style", self.default_style),
            "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
            "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None,
        }

    def save_user_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Ghi CHỈ các giọng người dùng ra JSON (mặc định ``user_voices_path``), ghi atomic.

        Gộp với nội dung đang có trên đĩa: giọng do tiến trình khác (CLI, web UI, API)
        thêm vào sau khi phiên này nạp file vẫn được giữ; chỉ giọng phiên này đã
        ``remove_voice`` mới bị bỏ. Trả về đường dẫn file.
        """
        import json
        import tempfile
        path = Path(path).expanduser() if path else self.user_voices_path
        presets: dict = {}
        if path.exists():
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                on_disk = on_disk.get("presets") if isinstance(on_disk, dict) else None
                if isinstance(on_disk, dict):
                    presets = {n: v for n, v in on_disk.items()
                               if isinstance(v, dict) and n not in self._builtin_voice_names}
            except Exception as e:  # noqa: BLE001
                logger.warning(f"⚠️ Không đọc được {path} để gộp, sẽ ghi lại từ bộ nhớ: {e}")
        for n in self._removed_user_voices:
            presets.pop(n, None)
        for n in list(self._user_voice_names):
            if n in self._preset_voices:
                presets[n] = self._voice_to_json(self._preset_voices[n])
        data = {"meta": {"note": "VieNeu user voices (cloned): speaker embedding + reference codes",
                         "count": len(presets)},
                "presets": presets}
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".user_voices_", suffix=".json", dir=str(path.parent))
        os.close(fd)
        try:
            Path(tmp).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        logger.info(f"💾 Saved {len(presets)} user voices → {path}")
        return str(path)

    def save_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export ALL current voices (built-in + user) to one JSON file.

        Mặc định ghi đè file asset của package — chỉ dùng khi cố ý xuất bộ giọng mới.
        Muốn lưu giọng vừa clone, dùng ``add_voice(..., save=True)`` / ``save_user_voices``.
        """
        import json
        path = Path(path) if path else self.builtin_voices_path()
        presets = {n: self._voice_to_json(v) for n, v in self._preset_voices.items()}
        data = {"meta": {"note": "v3 turbo voices: speaker embedding + reference codes"},
                "default_voice": self._default_voice, "presets": presets}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Saved {len(presets)} voices → {path}")
        return str(path)

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve the requested voice to ``(speaker_emb, ref_codes)``.

        Precedence: cloned ``ref_audio`` → preset ``voice`` (name or dict) → default preset.
        """
        if ref_audio is not None:
            import os
            clean_ref = self._preclean_reference_audio(ref_audio)
            try:
                return self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
            finally:
                if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                    try:
                        os.remove(clean_ref)
                    except Exception:
                        pass
        preset = None
        if isinstance(voice, str):
            preset = self._preset_voices.get(voice)
            if preset is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
        elif isinstance(voice, dict):
            preset = voice
        elif self._default_voice:
            preset = self._preset_voices[self._default_voice]
        if preset is None:
            raise ValueError("Provide a preset `voice` name or a `ref_audio` to clone.")
        codes = preset.get("codes") if use_ref_codes else None
        return preset["speaker_emb"], codes

    # ── Batched inference (GPU/PyTorch) ──────────────────────────────────────
    def _get_batch_engine(self):
        """Lazily build the static-batching engine. Returns ``None`` on CPU/ONNX.

        The batching runtime (``vieneu.v3_turbo_serve``) is pure PyTorch/CUDA, so it
        only applies to the ``pytorch`` backend. Importing it is deferred until the
        first batched call to keep the torch-free CPU path from touching torch.
        """
        if self.backend != "pytorch":
            return None
        if self._batch_engine is None:
            from .v3_turbo_serve import V3TurboBatchEngine
            self._batch_engine = V3TurboBatchEngine(self.engine)
        return self._batch_engine

    def _infer_chunks(
        self,
        chunks: List[str],
        speaker_emb: np.ndarray,
        ref_codes: Optional[np.ndarray],
        use_ref_codes: bool,
        batch_size: int,
        sampling: dict,
    ) -> List[np.ndarray]:
        """Synthesize one waveform per chunk (no join, no watermark).

        On the PyTorch/GPU backend with ``batch_size > 1`` and more than one chunk,
        the chunks are pushed through the static-batching engine in groups of
        ``batch_size`` (chunks share each forward step — the throughput win). Groups
        are formed over chunks SORTED by phoneme length (length bucketing): batching
        similar-length prompts together minimizes left-padding, which cuts wasted
        prefill compute and padding-induced numeric noise. On CPU (ONNX) or a single
        chunk, each chunk goes through the single-sequence engine sequentially.
        Output order always matches ``chunks``.
        """
        n = len(chunks)
        engine = self._get_batch_engine() if (batch_size > 1 and n > 1) else None

        if engine is None:
            wavs: List[np.ndarray] = []
            for chunk in chunks:
                ph = phonemize_text_with_emotions(chunk)
                wavs.append(self.engine.infer(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    **_cap_frames(sampling, max_expected_frames(ph)),
                ))
            return wavs

        phs = [phonemize_text_with_emotions(c) for c in chunks]
        order = sorted(range(n), key=lambda i: len(phs[i]))
        wavs = [None] * n
        for i in range(0, n, batch_size):
            idxs = order[i:i + batch_size]
            reqs = [{
                "phonemes": phs[j],
                "speaker_emb": speaker_emb, "ref_codes": ref_codes,
                "use_ref_codes": use_ref_codes,
            } for j in idxs]
            # Trần chung cho cả group = trần của row dài nhất — group đã được
            # bucket theo độ dài phoneme nên trần vẫn sát với từng row.
            group_cap = max(max_expected_frames(phs[j]) for j in idxs)
            for j, w in zip(idxs, engine.generate_batch(reqs, **_cap_frames(sampling, group_cap))):
                wavs[j] = w
        return wavs

    # ── Public API ───────────────────────────────────────────────────────────
    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua — style đã nằm trong ref code (luôn tự nhiên)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,   # GPU: trần chunk/forward (None → self.max_batch_size; 1 → tắt batch)
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 48 kHz waveform.

        ``style`` is deprecated and ignored (kept only so older code keeps running):
        the reading style comes from the reference voice, so output is always the
        natural style.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)

        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        # GPU gộp các chunk vào cùng forward; CPU/1-chunk chạy tuần tự (xem _infer_chunks).
        all_wavs = self._infer_chunks(
            chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Im lặng theo loại ranh giới: ngắt đoạn > hết câu > ngắt trong câu.
        final_wav = join_audio_chunks(
            all_wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps)
        )
        return self._apply_watermark(final_wav) if apply_watermark else final_wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        chunks = normalize_to_chunks_v3(text, max_chars=max_chars)
        # Prefer the engine's native frame-level streaming (low first-audio latency);
        # both the PyTorch and ONNX engines expose infer_stream. Fall back to one
        # full infer per chunk if not available.
        stream_fn = getattr(self.engine, "infer_stream", None)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            chunk_frames_cap = min(max_new_frames, max_expected_frames(ph))
            if stream_fn is not None:
                for sub in stream_fn(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=chunk_frames_cap, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                ):
                    if sub is None or len(sub) == 0:
                        continue
                    yield self._apply_watermark(sub) if apply_watermark else sub
            else:
                wav = self.engine.infer(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=chunk_frames_cap, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                )
                yield self._apply_watermark(wav) if apply_watermark else wav

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Synthesize many texts, returning one waveform each (same order as ``texts``).

        All texts share one voice (resolved once). On the PyTorch/GPU backend the
        chunks from *every* text are flattened and batched together in groups of
        ``batch_size`` — so even many short texts fill the batch and share forward steps.
        On CPU (ONNX) this runs sequentially. Output equals per-text ``infer`` (equivalent,
        not bit-identical to the single path, since sampling runs batched).
        """
        if not texts:
            return []

        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )

        # Cắt chunk từng text, nhớ chunk thuộc text nào (owner) và gaps để join lại.
        per_text_gaps: List[Any] = []
        flat_chunks: List[str] = []
        owner: List[int] = []
        for ti, t in enumerate(texts):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(t, max_chars=max_chars)
            per_text_gaps.append(gaps)
            for c in chunks:
                flat_chunks.append(c)
                owner.append(ti)

        empty = np.array([], dtype=np.float32)
        if not flat_chunks:
            return [empty for _ in texts]

        flat_wavs = self._infer_chunks(
            flat_chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Gom wav về đúng text (thứ tự flat_wavs khớp flat_chunks/owner), rồi join từng text.
        grouped: List[List[np.ndarray]] = [[] for _ in texts]
        for w, ti in zip(flat_wavs, owner):
            grouped[ti].append(w)

        results: List[np.ndarray] = []
        for ti in range(len(texts)):
            if not grouped[ti]:
                results.append(empty)
                continue
            joined = join_audio_chunks(
                grouped[ti], self.sample_rate, silence_ps=gaps_to_silence(per_text_gaps[ti])
            )
            results.append(self._apply_watermark(joined) if apply_watermark else joined)
        return results

    def close(self) -> None:
        self.engine = None
        self._batch_engine = None
