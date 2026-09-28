# VieNeu-TTS API server & giọng clone lưu bền

Tài liệu này mô tả hai tính năng đi kèm nhau:

1. **Giọng người dùng (user voices)**: clone một giọng từ audio mẫu rồi lưu bền, giọng đó
   xuất hiện trong danh sách chọn của SDK, web UI và API server.
2. **API server** (`apps/api_server.py`): HTTP API để ứng dụng khác gọi TTS bằng bất kỳ
   giọng nào trong danh sách, kèm endpoint tương thích OpenAI Audio API.

Cả hai chạy trên đường CPU/ONNX (torch-free) mặc định của v3 Turbo, 48 kHz.

---

## 1. Giọng người dùng lưu ở đâu

| | |
|---|---|
| File | `~/.cache/vieneu/user_voices.json` (Windows: `C:\Users\<user>\.cache\vieneu\user_voices.json`) |
| Đổi vị trí | đặt biến môi trường `VIENEU_USER_VOICES=/duong/dan/user_voices.json` hoặc `Vieneu(user_voices="...")` |
| Nội dung | speaker embedding 192 chiều + reference codes của mỗi giọng (không chứa audio gốc) |
| Giọng có sẵn | vẫn nằm trong package (`assets/voices_v3_turbo.json`), không bị ghi đè |

Quy tắc: tên giọng người dùng **không được trùng** tên giọng có sẵn (Adam, Minh Đức, ...).
Trùng tên giọng người dùng cũ thì thay thế. File hỏng hoặc thiếu chỉ gây cảnh báo, không làm
model không nạp được.

Muốn đem giọng lên server khác: copy file JSON này sang máy đích (hoặc gọi `POST /voices`
trên server với cùng file audio mẫu).

### Thêm giọng bằng dòng lệnh

```bash
uv run --frozen python -m apps.voice_tool add clone_A outputs/clone/mau_A.wav \
    --description "Nam · Giọng từ video X" --test outputs/thu.wav
uv run --frozen python -m apps.voice_tool list
uv run --frozen python -m apps.voice_tool remove clone_A
uv run --frozen python -m apps.voice_tool path
```

Audio mẫu: WAV/MP3/FLAC/OGG, một người nói, 3 đến 8 giây, không nhạc nền. Engine tự cắt
khoảng lặng hai đầu và khử nhiễu (tắt bằng `--no-denoise` nếu audio đã sạch).

### Thêm giọng bằng SDK

```python
from vieneu import Vieneu
tts = Vieneu()
tts.add_voice("clone_A", "mau_A.wav", description="Nam · Giọng từ video X", save=True)

# Các phiên sau tự có giọng này
tts = Vieneu()
audio = tts.infer("Xin chào.", voice="clone_A")
tts.save(audio, "out.wav")

tts.list_voices()          # [{"id", "label", "description", "gender", "custom"}, ...]
tts.is_user_voice("clone_A")   # True
tts.remove_voice("clone_A", save=True)
```

### Trong web UI

Bấm **Tải Model**, giọng người dùng xuất hiện trong dropdown **Giọng mẫu** cùng 20 giọng có sẵn.
Nếu web UI đang chạy khi bạn thêm giọng bằng lệnh khác, bấm Tải Model lại để nạp file mới.

---

## 2. Chạy API server

```bash
# cục bộ
uv run --frozen python -m apps.api_server              # http://127.0.0.1:8000/docs

# trên VPS: mở ra mạng + bắt buộc API key + số luồng = số vCPU
VIENEU_API_KEY=doi-thanh-chuoi-bi-mat VIENEU_THREADS=4 \
uv run --frozen python -m apps.api_server --host 0.0.0.0 --port 8000
```

Lệnh `vieneu-api` cũng có sau khi cài lại project (`uv sync --frozen`), nhưng
`python -m apps.api_server` luôn chạy được.

Model được nạp **một lần** lúc khởi động (5 đến 10 giây, nhanh hơn nếu đặt `HF_HUB_OFFLINE=1`
sau lần tải đầu). Các request sinh giọng được xếp hàng và chạy tuần tự trên một engine.

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIENEU_API_HOST` | `127.0.0.1` | `0.0.0.0` để nhận kết nối từ ngoài |
| `VIENEU_API_PORT` | `8000` | cổng |
| `VIENEU_API_KEY` | trống (tắt) | bật xác thực: `Authorization: Bearer <key>` hoặc `X-API-Key: <key>` |
| `VIENEU_DEFAULT_VOICE` | giọng mặc định của model (Adam) | giọng dùng khi request không có `voice` |
| `VIENEU_THREADS` | `cpu_count // 2` (tối đa 8) | số luồng ONNX; trên VPS đặt bằng số vCPU |
| `VIENEU_API_MAX_CHARS` | `3000` | giới hạn độ dài văn bản |
| `VIENEU_API_MAX_QUEUE` | `16` | số request được xếp hàng chờ engine; vượt quá trả `429` kèm `Retry-After` |
| `VIENEU_API_CORS_ORIGINS` | trống (tắt) | danh sách origin cách nhau bằng phẩy nếu gọi thẳng từ trình duyệt |
| `VIENEU_USER_VOICES` | `~/.cache/vieneu/user_voices.json` | file giọng người dùng |

Engine chỉ sinh một request tại một thời điểm; các request khác chờ trên event loop, nên
`/health` và `/voices` vẫn trả lời khi engine đang bận. Định dạng `mp3`, `opus`, `aac` cần `ffmpeg`
trên PATH của server (`apt install ffmpeg`), nếu không có thì trả `415`; `wav`, `pcm`, `flac` luôn dùng được.

---

## 3. Endpoints

Swagger UI tại `/docs` (trang tài liệu và `/openapi.json` luôn công khai). Mọi endpoint dữ liệu trừ `/`
và `/health` đều cần API key khi đã bật.

### `GET /health`

```json
{"status": "ok", "backend": "onnx", "sample_rate": 48000, "voices": 21, "auth": false,
 "busy": false, "queue": 0, "formats": ["wav", "pcm", "flac", "mp3", "opus", "aac"]}
```

### `GET /voices`

```json
{"default": "Adam",
 "voices": [{"id": "Adam", "label": "Adam — Nam · Nam · Giọng đọc tự nhiên", "description": "...", "gender": "male", "custom": false},
            {"id": "clone_A", "label": "clone_A — Giọng clone từ video", "description": "...", "gender": "", "custom": true}]}
```

### `POST /tts` → audio

Body JSON:

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `text` | bắt buộc | tối đa `VIENEU_API_MAX_CHARS` ký tự |
| `voice` | giọng mặc định | ID trong `GET /voices`, ví dụ `clone_A` |
| `temperature` | 0.8 | 0.05 đến 2.0, thấp hơn thì ổn định hơn |
| `top_k` / `top_p` | 25 / 0.95 | |
| `repetition_penalty` | 1.2 | |
| `max_chars` | 256 | ngưỡng cắt chunk |
| `sample_rate` | 48000 | resample đầu ra, ví dụ 24000 hoặc 16000 |
| `format` | `wav` | `wav` (PCM16, `audio/wav`), `pcm` (raw PCM16 little-endian mono, `audio/pcm`), `flac`, `mp3`, `opus` (trong ogg), `aac` |
| `watermark` | true | |

Header phản hồi: `X-Voice` (tên giọng đã percent-encode, ví dụ `Minh%20%C4%90%E1%BB%A9c`),
`X-Sample-Rate`, `X-Audio-Seconds`, `X-Generation-Seconds`, và `X-PCM-Format: s16le mono` với `pcm`.

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào, đây là giọng clone.", "voice": "clone_A"}' \
  -o out.wav
```

Trên PowerShell dùng file JSON để tránh lỗi escape dấu nháy:

```powershell
'{"text": "Xin chào, đây là giọng clone.", "voice": "clone_A"}' | Set-Content -Encoding utf8 req.json
curl.exe -X POST http://127.0.0.1:8000/tts -H "Content-Type: application/json" --data-binary "@req.json" -o out.wav
```

Python:

```python
import requests
r = requests.post("http://127.0.0.1:8000/tts",
                  json={"text": "Xin chào, đây là giọng clone.", "voice": "clone_A", "sample_rate": 24000},
                  headers={"X-API-Key": "..."}, timeout=120)
r.raise_for_status()
open("out.wav", "wb").write(r.content)
print(r.headers["X-Audio-Seconds"], "giây audio")
```

Lỗi: `404` giọng không tồn tại (kèm danh sách `available`), `422` văn bản trống, quá dài hoặc không có
gì để đọc (chỉ emoji/ký hiệu), `415` định dạng cần ffmpeg mà server không có, `429` hàng chờ đầy
(đọc `Retry-After` rồi thử lại), `401` sai API key, `503` model chưa nạp xong.

### `POST /tts/stream` → WAV stream

Cùng body như `/tts` (không có `sample_rate`/`format`). Trả về WAV 48 kHz theo chunk, âm thanh
đầu tiên tới sau khoảng 300 ms, phù hợp phát ngay trên trình duyệt hoặc app.

```bash
curl -N -X POST http://127.0.0.1:8000/tts/stream -H "Content-Type: application/json" \
  -d '{"text": "Xin chào các bạn.", "voice": "clone_A"}' -o stream.wav
```

Header WAV của stream ghi độ dài rất lớn (chưa biết trước). Player phát trực tiếp thì không sao;
muốn lưu file chuẩn thì ghi lại header sau khi nhận xong, hoặc dùng `/tts`.

### `POST /v1/audio/speech` (tương thích OpenAI)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="doi-thanh-chuoi-bi-mat")
audio = client.audio.speech.create(model="vieneu-v3-turbo", voice="clone_A",
                                   input="Xin chào, đây là giọng clone.", response_format="wav")
audio.write_to_file("out.wav")
```

`model` và `speed` được chấp nhận nhưng bỏ qua; `response_format` nhận `wav`, `pcm`, `flac`, và `mp3`,
`opus`, `aac` khi server có ffmpeg (nếu client gửi `mp3` mà server không có ffmpeg sẽ nhận `415`).
Lỗi trên các đường `/v1/*` trả về theo dạng OpenAI `{"error": {"message": ..., "type": ...}}`.
`GET /v1/models` trả về một model `vieneu-v3-turbo`.

### `POST /voices` (thêm giọng qua API, multipart)

```bash
curl -X POST http://127.0.0.1:8000/voices \
  -F "name=clone_B" -F "description=Nữ · Giọng từ podcast" -F "file=@mau_B.wav"
```

Giọng được đăng ký vào server đang chạy **và** lưu vào file giọng người dùng (`"persisted": true`
trong phản hồi). Trả về `409` nếu tên trùng giọng có sẵn, `415` định dạng không hỗ trợ, `413` file
quá 25 MB, `422` audio ngắn hơn 1 giây, dài hơn 60 giây hoặc không đọc được.

### `DELETE /voices/{name}`

Chỉ xoá được giọng người dùng (`403` với giọng có sẵn).

---

## 4. Ghi chú khi đưa lên VPS

- Chọn gói vCPU dedicated, 4 vCPU và 8 GB RAM là đủ cho nhóm nhỏ. Đặt `VIENEU_THREADS` bằng số vCPU.
- Luôn đặt `VIENEU_API_KEY` khi `--host 0.0.0.0`. Nếu cần HTTPS, đặt Nginx hoặc Caddy phía trước.
- Mỗi tiến trình chỉ sinh một request tại một thời điểm. Cần phục vụ nhiều người cùng lúc thì
  chạy nhiều tiến trình trên các cổng khác nhau sau load balancer.
- Chạy nền bằng systemd. Đặt `User=` là tài khoản đã cài và đã đăng ký giọng, vì file giọng
  người dùng và cache HuggingFace nằm trong home của tài khoản đó (`~/.cache/vieneu`, `~/.cache/huggingface`):

```ini
[Unit]
Description=VieNeu-TTS API
After=network.target

[Service]
User=vieneu
WorkingDirectory=/opt/VieNeu-TTS
Environment=VIENEU_API_KEY=doi-thanh-chuoi-bi-mat
Environment=VIENEU_THREADS=4
Environment=HF_HUB_OFFLINE=1
ExecStart=/opt/VieNeu-TTS/.venv/bin/python -m apps.api_server --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
