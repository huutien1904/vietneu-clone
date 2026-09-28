# Hướng dẫn tích hợp VieNeu-TTS (Python SDK) vào dự án của bạn

Tài liệu này mô tả cách nhúng **VieNeu-TTS** trực tiếp vào một dự án Python bằng SDK `vieneu`
(Cách 1 — chạy local, cùng tiến trình, **không cần server**). Mặc định SDK dùng **v3 Turbo, 48 kHz**,
đường CPU/ONNX (torch-free) — phù hợp nhất cho hầu hết ứng dụng.

> Kiểm chứng theo source: `src/vieneu/factory.py`, `src/vieneu/v3turbo.py`, `src/vieneu/base.py`.

---

## 1. Cài đặt

```bash
pip install vieneu
```

- **CPU (mặc định)** — torch-free, chạy v3 Turbo qua ONNX Runtime. Đa số dùng bản này.
  Backbone chạy `int8` mặc định (~1.6× nhanh, ~4× nhỏ, chất lượng gần như giữ nguyên).
- **GPU (CUDA)** — chỉ khi cần batch/throughput cao cho văn bản dài:
  ```bash
  pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install "transformers==4.57.6"
  pip install vieneu
  ```

> **Lần chạy đầu** sẽ tự tải model weights từ HuggingFace về cache (`~/.cache/huggingface`).
> Sau đó các lần sau load từ cache, nhanh.

---

## 2. Khởi tạo — chỉ tạo **một lần** rồi tái sử dụng

`Vieneu()` load model vào RAM (nặng). **Tạo một instance duy nhất** khi khởi động ứng dụng
và dùng lại cho mọi request. **Không** tạo mới mỗi lần gọi.

```python
from vieneu import Vieneu

# Mặc định = v3 Turbo 48 kHz. backend="auto" → ONNX trên CPU, PyTorch trên GPU.
tts = Vieneu()

# Muốn ép chạy CPU/ONNX (đường streaming, low-latency), kể cả trên máy có GPU:
# tts = Vieneu(backend="onnx")

# Muốn chất lượng tối đa trên CPU (chậm hơn):
# tts = Vieneu(precision="fp32")
```

### Tham số khởi tạo hay dùng (`Vieneu(...)`)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `backend` | `"auto"` | `"auto"` → ONNX/CPU hoặc PyTorch/GPU tự động; ép bằng `"onnx"` \| `"pytorch"` |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` |
| `precision` | `"int8"` | Chỉ áp cho CPU/ONNX: `"int8"` (nhanh) \| `"fp32"` (chất lượng tối đa). Bỏ qua trên GPU |
| `threads` | `0` | Số intra-op threads cho CPU/ONNX; `0` = tự động (~số nhân vật lý, tối đa 8) |
| `max_batch_size` | `32` | Chỉ GPU: trần số chunk gộp vào một forward |

Sau khi tạo: `tts.sample_rate == 48000`, `tts.backend` là `"onnx"` hoặc `"pytorch"`.

---

## 3. Sinh giọng cơ bản (`infer`)

`infer(...)` trả về **một mảng `numpy.ndarray` float32 @ 48 kHz** (dạng sóng mono).

```python
# 1) Dùng giọng preset dựng sẵn (không cần clip mẫu)
audio = tts.infer("Xin chào các bạn, đây là VieNeu-TTS.", voice="Adam")
tts.save(audio, "output.wav")     # ghi WAV 48 kHz

# 2) Tự lấy dữ liệu sóng để đưa vào pipeline khác (bytes, phát trực tiếp, v.v.)
import numpy as np
print(audio.dtype, audio.shape, len(audio) / tts.sample_rate, "giây")
```

### Chữ ký `infer` (các tham số hay dùng)

```python
tts.infer(
    text: str,                       # văn bản cần đọc (bắt buộc)
    ref_audio: str | None = None,    # đường dẫn clip mẫu để CLONE giọng (xem mục 6)
    voice:  str | dict | None = None,# tên giọng preset / giọng đã add_voice, hoặc dict giọng
    denoise: bool = True,            # khử nhiễu clip mẫu trước khi clone
    temperature: float = 0.8,        # độ ngẫu nhiên (thấp = ổn định hơn)
    top_k: int = 25,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_chars: int = 256,            # ngưỡng cắt chunk theo ký tự
    apply_watermark: bool = True,    # đóng dấu bản quyền ẩn vào audio
    batch_size: int | None = None,   # chỉ GPU: trần chunk/forward
) -> np.ndarray
```

> **Lưu ý:** tham số `style` đã **deprecated** trên v3 Turbo — truyền vào cũng bị bỏ qua.
> Phong cách đọc nằm trong chính giọng (preset hoặc clip mẫu). Code mới nên bỏ hẳn `style`.

### Cảm xúc / phi lời nói (thử nghiệm)

Chèn tag trực tiếp vào text: `[cười]`, `[thở dài]`, `[hắng giọng]`.

```python
audio = tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Adam")
```

---

## 4. Danh sách giọng preset

```python
voices = tts.list_preset_voices()     # -> List[tuple[label, voice_id]]
for label, voice_id in voices:
    print(f"{label}  ->  voice='{voice_id}'")
```

v3 Turbo có **20 giọng** theo 3 vùng miền, ví dụ:
- **Bắc**: Minh Đức, Phạm Tuyên, Trúc Ly, Mai Anh, Quỳnh Anh
- **Trung**: Quang Sơn, Ngọc Trân
- **Nam**: Adam *(mặc định)*, Xuân Vĩnh, Thái Sơn, Thùy Dung, Mỹ Duyên

> Luôn lấy danh sách chính xác bằng `list_preset_voices()` tại runtime thay vì hardcode.

---

## 5. Streaming realtime (`infer_stream`)

Trả về **generator** phát từng chunk `np.float32 @ 48 kHz` ngay khi có (âm thanh bắt đầu ~300 ms,
RTF < 1 trên CPU) — lý tưởng cho ứng dụng tương tác. Nên ép `backend="onnx"` cho streaming.

```python
tts = Vieneu(backend="onnx")
for chunk in tts.infer_stream("Xin chào, đây là chế độ phát trực tiếp.", voice="Adam"):
    # chunk: np.float32 @ 48 kHz — phát ra loa hoặc ghi ra socket ngay khi tới
    play(chunk)   # ví dụ: sounddevice / ghi vào response stream
```

---

## 6. Voice cloning — nhân bản giọng từ clip mẫu

Cần clip mẫu **3–8 giây**. Clip được tự khử nhiễu và cắt ngắn trước khi clone (giữ `denoise=True`
trừ khi clip đã sạch). *(Tính năng clone chỉ có từ v3 trở lên.)*

```python
audio = tts.infer(
    "Đây là giọng được nhân bản tức thì.",
    ref_audio="my_voice.wav",   # clip mẫu 3-8s
    denoise=True,
)
tts.save(audio, "cloned.wav")
```

### Lưu & tái sử dụng giọng đã clone (`add_voice`)

Đăng ký một lần rồi gọi bằng tên như giọng preset:

```python
# name, ref_audio, [denoise], [description], [gender], [save]
tts.add_voice("Giọng của tôi", "my_voice.wav", denoise=True)

audio = tts.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Lưu ra file để phiên sau load lại
tts.save_voices()                 # ghi vào file voices mặc định
# tts.remove_voice("Giọng của tôi")

# Clip đã tự làm sạch → bỏ qua denoise cho nhanh
tts.add_voice("Giọng sạch", "already_clean.wav", denoise=False)
```

### Chỉ khử nhiễu một clip (không sinh audio)

```python
wav, sr = tts.denoise("noisy.wav", out_path="clean.wav")   # 44.1 kHz mono
```

> `denoise`, `add_voice` và cloning chạy trên **mọi backend**, kể cả bản CPU/ONNX torch-free.

---

## 7. Sinh hàng loạt (`infer_batch`)

Nhiều text, trả về danh sách sóng cùng thứ tự. Trên **GPU** các chunk của mọi text được gộp batch
(tăng throughput mạnh cho văn bản dài/nhiều). Trên **CPU** chạy tuần tự (vẫn hoạt động, không lỗi).

```python
texts = [
    "Câu thứ nhất cần đọc.",
    "Câu thứ hai cần đọc.",
    "Câu thứ ba cần đọc.",
]
audios = tts.infer_batch(texts, voice="Adam")   # -> List[np.ndarray]
for i, a in enumerate(audios):
    tts.save(a, f"batch_{i}.wav")
```

---

## 8. Mẫu tích hợp gợi ý (singleton service)

Đóng gói thành một service khởi tạo một lần, dùng lại khắp nơi:

```python
# tts_service.py
from functools import lru_cache
from vieneu import Vieneu

@lru_cache(maxsize=1)
def get_tts() -> Vieneu:
    # Chỉ chạy 1 lần cho cả tiến trình
    return Vieneu()   # hoặc Vieneu(backend="onnx") cho streaming

def synthesize(text: str, voice: str = "Adam", out_path: str | None = None):
    tts = get_tts()
    audio = tts.infer(text, voice=voice)
    if out_path:
        tts.save(audio, out_path)
    return audio          # np.float32 @ 48 kHz
```

```python
# chỗ dùng
from tts_service import synthesize
synthesize("Xin chào từ dự án của tôi.", voice="Adam", out_path="hello.wav")
```

> **Lưu ý luồng (thread-safety):** một instance `Vieneu` **không đảm bảo an toàn khi gọi song song
> nhiều luồng**. Với web server đa luồng, hãy **serialize** truy cập (một khóa/`Lock`) hoặc dùng
> một hàng đợi/worker chuyên xử lý TTS, để tránh nhiều request cùng gọi `infer` một lúc trên cùng model.

---

## 9. Bảng tra nhanh API

| Việc cần làm | Gọi |
|---|---|
| Khởi tạo model (1 lần) | `Vieneu()` / `Vieneu(backend="onnx")` |
| Đọc text → sóng 48 kHz | `tts.infer(text, voice="Adam")` |
| Ghi ra file WAV | `tts.save(audio, "out.wav")` |
| Streaming realtime | `for c in tts.infer_stream(text, voice=...)` |
| Nhiều text một lượt | `tts.infer_batch([...], voice=...)` |
| Liệt kê giọng preset | `tts.list_preset_voices()` |
| Clone từ clip | `tts.infer(text, ref_audio="clip.wav")` |
| Lưu giọng để tái dùng | `tts.add_voice(name, "clip.wav")` → `tts.save_voices()` |
| Chỉ khử nhiễu clip | `tts.denoise("noisy.wav", out_path="clean.wav")` |

**Định dạng output:** `np.ndarray`, `dtype=float32`, mono, **48 000 Hz**.

---

## 10. Lưu ý cho Windows

Nếu bạn `print(...)` nhãn giọng/text tiếng Việt ra console Windows (mặc định cp1252) có thể gặp
`UnicodeEncodeError`. Đây là lỗi *in ra console*, **không phải lỗi SDK** — audio vẫn sinh bình thường.
Khắc phục bằng cách ép UTF-8:

```bash
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
```

hoặc trong code: `sys.stdout.reconfigure(encoding="utf-8")`.

> **Đã kiểm chứng thực tế** (Windows, CPU/ONNX): load model ~8–11s (lần đầu), `infer` một câu ~2.2s
> audio sinh trong ~1.2s (nhanh hơn realtime). Đường CPU/ONNX không cần GPU, không cần PyTorch.
