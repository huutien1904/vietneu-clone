"""
Quản lý giọng người dùng (clone → lưu bền) cho VieNeu-TTS v3 Turbo.
====================================================================
Giọng lưu trong ``~/.cache/vieneu/user_voices.json`` (hoặc ``$VIENEU_USER_VOICES``)
và tự xuất hiện trong SDK (``voice="tên"``), web UI (dropdown Giọng mẫu sau khi
tải model) và API server (``GET /voices``).

    uv run --frozen python -m apps.voice_tool add clone_A outputs/clone/mau_A.wav --description "Nam · Giọng từ video"
    uv run --frozen python -m apps.voice_tool list
    uv run --frozen python -m apps.voice_tool remove clone_A
    uv run --frozen python -m apps.voice_tool path

``add`` cần nạp model (để trích speaker embedding, ~5-10 s); ``list``/``remove``/``path``
chỉ đọc/ghi file JSON nên chạy tức thì.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _user_path() -> Path:
    from vieneu.v3turbo import default_user_voices_path
    return default_user_voices_path()


def _builtin_names() -> list:
    from vieneu.v3turbo import V3TurboVieNeuTTS
    p = V3TurboVieNeuTTS.builtin_voices_path()
    if not p.exists():
        return []
    try:
        return list((json.loads(p.read_text(encoding="utf-8")).get("presets") or {}).keys())
    except Exception:  # noqa: BLE001
        return []


def _read_user(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("presets") or {}
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Không đọc được {path}: {e}")
        return {}


def _write_user(path: Path, presets: dict) -> None:
    data = {"meta": {"note": "VieNeu user voices (cloned): speaker embedding + reference codes",
                     "count": len(presets)}, "presets": presets}
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
            except Exception:  # noqa: BLE001
                pass


def cmd_add(args) -> int:
    ref = Path(args.ref_audio)
    if not ref.exists():
        print(f"❌ Không thấy file audio mẫu: {ref}")
        return 1
    if args.name in _builtin_names():
        print(f"❌ '{args.name}' là giọng có sẵn, hãy chọn tên khác.")
        return 1
    from vieneu import Vieneu
    t0 = time.time()
    print("⏳ Đang nạp model để trích đặc trưng giọng...")
    tts = Vieneu()
    print(f"   model sẵn sàng sau {time.time() - t0:.1f}s (backend={tts.backend})")
    tts.add_voice(args.name, str(ref), denoise=not args.no_denoise,
                  description=args.description, gender=args.gender, save=True)
    print(f"✅ Đã lưu giọng '{args.name}' → {tts.user_voices_path}")
    if args.test:
        out = Path(args.test)
        audio = tts.infer("Xin chào, đây là giọng vừa được thêm vào danh sách.", voice=args.name)
        tts.save(audio, out)
        print(f"🔊 Đã sinh thử → {out} ({len(audio) / tts.sample_rate:.2f}s)")
    return 0


def cmd_list(args) -> int:
    path = _user_path()
    user = _read_user(path)
    builtin = _builtin_names()
    print(f"📁 File giọng người dùng: {path} ({'có' if path.exists() else 'chưa có'})")
    print(f"🎙️  Giọng có sẵn: {len(builtin)}  |  Giọng người dùng: {len(user)}")
    for name, v in user.items():
        desc = v.get("description") or ""
        print(f"   - {name}" + (f" — {desc}" if desc else ""))
    if args.all:
        for name in builtin:
            print(f"   · {name} (có sẵn)")
    return 0


def cmd_remove(args) -> int:
    path = _user_path()
    user = _read_user(path)
    if args.name not in user:
        print(f"❌ Không có giọng người dùng tên '{args.name}' trong {path}")
        return 1
    user.pop(args.name)
    _write_user(path, user)
    print(f"🗑️  Đã xoá '{args.name}'. Còn {len(user)} giọng người dùng.")
    return 0


def cmd_path(args) -> int:
    print(_user_path())
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Quản lý giọng người dùng VieNeu-TTS (clone đã lưu)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Đăng ký giọng mới từ file audio mẫu và lưu bền")
    a.add_argument("name", help="ID giọng, vd clone_A")
    a.add_argument("ref_audio", help="File audio mẫu 3-8 giây (.wav/.mp3/.flac/.ogg)")
    a.add_argument("--description", default="", help="Mô tả hiển thị, vd 'Nam · Giọng từ video X'")
    a.add_argument("--gender", default="", help="male/female (tuỳ chọn)")
    a.add_argument("--no-denoise", action="store_true", help="Không khử nhiễu audio mẫu")
    a.add_argument("--test", metavar="OUT.wav", help="Sinh thử một câu bằng giọng mới và lưu ra file")
    a.set_defaults(fn=cmd_add)

    l = sub.add_parser("list", help="Liệt kê giọng người dùng")
    l.add_argument("--all", action="store_true", help="Liệt kê cả giọng có sẵn")
    l.set_defaults(fn=cmd_list)

    r = sub.add_parser("remove", help="Xoá giọng người dùng")
    r.add_argument("name")
    r.set_defaults(fn=cmd_remove)

    sub.add_parser("path", help="In đường dẫn file giọng người dùng").set_defaults(fn=cmd_path)

    args = p.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
