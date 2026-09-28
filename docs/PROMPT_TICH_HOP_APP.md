# Prompt tích hợp VieNeu-TTS API vào ứng dụng (giọng clone_A)

Dán đoạn dưới đây cho trợ lý AI hoặc lập trình viên đang làm ứng dụng của bạn.
Thay `<HOST>` bằng IP/tên miền server và `<API_KEY>` bằng giá trị `VIENEU_API_KEY` (bỏ phần key nếu server không bật xác thực).

Giá trị khi chạy trên chính máy này (server khởi động bằng `uv run --frozen python -m apps.api_server`):
`<HOST>` = `127.0.0.1` nếu app chạy cùng máy, hoặc `192.168.0.101` nếu app ở máy/điện thoại khác trong
mạng LAN (khi đó khởi động server với `--host 0.0.0.0 --api-key <chuoi>`); chưa bật API key thì bỏ header key.

---

Tôi có một server text-to-speech tiếng Việt (VieNeu-TTS v3 Turbo) đang chạy tại `http://<HOST>:8000`.
Hãy viết cho ứng dụng của tôi một module client gọi server này để chuyển văn bản thành giọng nói bằng giọng có id **`clone_A`** (giọng đã clone sẵn trên server). Yêu cầu và đặc tả API như sau.

## Yêu cầu chức năng

1. Hàm `synthesize(text, voice="clone_A") -> bytes` trả về file WAV; hàm `synthesize_to_file(text, path, voice="clone_A")` lưu ra đĩa.
2. Hàm `list_voices()` trả về danh sách giọng từ server, để UI cho người dùng chọn; mặc định chọn `clone_A`.
3. Xử lý lỗi rõ ràng: 401 (sai API key), 404 (giọng không tồn tại, server trả kèm danh sách `available`), 422 (văn bản trống hoặc quá dài), 503 (server đang nạp model, thử lại sau vài giây), lỗi mạng/timeout.
4. Văn bản dài hơn 3000 ký tự phải được cắt theo câu hoặc đoạn rồi gọi nhiều lần, ghép audio theo thứ tự.
5. Các lời gọi phải chạy tuần tự hoặc qua hàng đợi phía client. Server sinh từng request một, gửi song song chỉ làm mọi request chờ lâu hơn.
6. Timeout mỗi request tối thiểu 120 giây. Tốc độ server khoảng 2 giây audio mỗi giây xử lý trên CPU.
7. API key đọc từ biến môi trường, không hard-code. Gửi bằng header `X-API-Key: <API_KEY>` hoặc `Authorization: Bearer <API_KEY>`.
8. Nếu ứng dụng cần phát ngay khi đang sinh (chatbot, trợ lý thoại), dùng endpoint stream (mục 3) và phát dần từng chunk.

## Đặc tả API

Base URL: `http://<HOST>:8000`. Tài liệu Swagger tại `/docs`.

### 1. `GET /health`
Không cần key. Trả về `{"status": "ok" | "loading", "backend": "onnx", "sample_rate": 48000, "voices": 21, "auth": true|false}`. Chỉ gọi TTS khi `status == "ok"`.

### 2. `GET /voices`
Trả về `{"default": "Adam", "voices": [{"id": "clone_A", "label": "clone_A — ...", "description": "...", "gender": "", "custom": true}, ...]}`. Giá trị dùng để gọi TTS là trường `id`.

### 3. `POST /tts`  (JSON → audio)
Body:
```json
{
  "text": "Xin chào, đây là giọng clone.",
  "voice": "clone_A",
  "sample_rate": 48000,
  "format": "wav",
  "temperature": 0.8
}
```
- `text` bắt buộc, tối đa 3000 ký tự. `voice` mặc định là giọng mặc định của server nếu bỏ trống, nên luôn truyền `"clone_A"`.
- `sample_rate` tuỳ chọn (8000 đến 48000), ví dụ 24000 hoặc 16000 để file nhỏ hơn. Mặc định 48000.
- `format`: `"wav"` (WAV PCM 16-bit mono, `audio/wav`), `"pcm"` (raw PCM16 little-endian mono, `audio/pcm`), `"flac"`, hoặc `"mp3"`/`"opus"`/`"aac"` nếu server có ffmpeg (không có thì trả `415`, khi đó dùng `wav`).
- Header phản hồi: `X-Voice` (tên giọng đã percent-encode), `X-Sample-Rate`, `X-Audio-Seconds` (độ dài audio), `X-Generation-Seconds` (thời gian sinh).
- Mã lỗi thêm: `429` khi hàng chờ của server đầy, đọc header `Retry-After` (giây) rồi thử lại.

### 4. `POST /tts/stream`  (JSON → WAV stream)
Body: `{"text": "...", "voice": "clone_A"}`. Trả về `audio/wav` 48 kHz theo chunk, audio đầu tiên tới sau khoảng 0.3 đến 0.5 giây. Header WAV ghi độ dài rất lớn vì chưa biết trước; phát trực tiếp thì bình thường, muốn lưu file chuẩn thì dùng `/tts`.

### 5. `POST /v1/audio/speech`  (tương thích OpenAI Audio API)
Body: `{"model": "vieneu-v3-turbo", "input": "...", "voice": "clone_A", "response_format": "wav"}`. Nếu ứng dụng đã dùng SDK OpenAI thì chỉ cần đổi `base_url` thành `http://<HOST>:8000/v1` và `voice="clone_A"`; `response_format` nhận `wav`, `pcm`, `flac`, và `mp3`/`opus`/`aac` khi server có ffmpeg. Lỗi trên `/v1/*` trả về dạng OpenAI `{"error": {"message": ...}}`.

## Ví dụ tham chiếu

Python:
```python
import os, requests

BASE = os.getenv("VIENEU_URL", "http://<HOST>:8000")
HEADERS = {"X-API-Key": os.getenv("VIENEU_API_KEY", "")}

def synthesize(text: str, voice: str = "clone_A", sample_rate: int | None = None) -> bytes:
    body = {"text": text, "voice": voice}
    if sample_rate:
        body["sample_rate"] = sample_rate
    r = requests.post(f"{BASE}/tts", json=body, headers=HEADERS, timeout=120)
    if r.status_code == 404:
        raise ValueError(f"Giọng không tồn tại. Có sẵn: {r.json()['detail']['available']}")
    r.raise_for_status()
    return r.content

open("out.wav", "wb").write(synthesize("Xin chào, đây là giọng clone A."))
```

JavaScript (Node 18+ hoặc trình duyệt):
```js
const BASE = process.env.VIENEU_URL ?? "http://<HOST>:8000";
const HEADERS = { "Content-Type": "application/json", "X-API-Key": process.env.VIENEU_API_KEY ?? "" };

export async function synthesize(text, voice = "clone_A") {
  const res = await fetch(`${BASE}/tts`, {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({ text, voice }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!res.ok) throw new Error(`TTS ${res.status}: ${await res.text()}`);
  return new Uint8Array(await res.arrayBuffer());   // WAV 48 kHz mono PCM16
}
```

Dùng SDK OpenAI:
```python
from openai import OpenAI
client = OpenAI(base_url="http://<HOST>:8000/v1", api_key="<API_KEY>")
resp = client.audio.speech.create(model="vieneu-v3-turbo", voice="clone_A",
                                  input="Xin chào, đây là giọng clone A.", response_format="wav")
resp.write_to_file("out.wav")
```

curl:
```bash
curl -X POST http://<HOST>:8000/tts -H "Content-Type: application/json" -H "X-API-Key: <API_KEY>" \
  -d '{"text": "Xin chào, đây là giọng clone A.", "voice": "clone_A"}' -o out.wav
```

Hãy viết module theo ngôn ngữ và framework của ứng dụng hiện tại, kèm test gọi thử `GET /health` và một lần `POST /tts` với giọng `clone_A`.
