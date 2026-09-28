# Deploy VieNeu-TTS API lên Modal.com (serverless CPU)

File deploy: [`deploy/modal_app.py`](../deploy/modal_app.py). Nó gói API server
(`apps/api_server.py`), model v3 Turbo ONNX int8 (tải sẵn vào image lúc build), ffmpeg và
giọng người dùng (clone_A, ...) thành một endpoint HTTPS. Container chỉ chạy khi có request và
tự tắt sau `scaledown_window` giây rảnh, nên chi phí tính theo giây thực chạy.

Số liệu trong tài liệu này lấy từ trang giá và tài liệu chính thức của Modal ngày 27/09/2026.

## 1. Có chạy miễn phí được không

**Có, trong hạn mức 30 USD compute mỗi tháng của gói Starter** (gói 0 USD/tháng, credit cấp
lại hằng tháng; trang giá không nói credit chưa dùng có cộng dồn hay không, nên coi như không).
Đăng ký bằng GitHub, Google hoặc SSO. Tài liệu billing của Modal ghi rằng tài khoản cần có
phương thức thanh toán, nên nhiều khả năng phải thêm thẻ; chỉ bị trừ tiền khi dùng vượt 30 USD.

Với cấu hình mặc định trong file deploy (2 nhân vật lý = 4 vCPU, 4 GiB RAM), 30 USD tương đương
khoảng **237 giờ container chạy mỗi tháng**. Một request câu ngắn tốn 3 đến 6 giây container, một lần
khởi động lạnh tốn thêm 10 đến 15 giây. Vài nghìn request mỗi tháng vẫn nằm trong hạn mức.

Ba điểm phải chấp nhận:

- **Khởi động lạnh**: request đầu tiên sau khi container tắt chờ thêm 10 đến 15 giây (khởi động
  container + nạp model). Các request trong 2 phút tiếp theo (mặc định) chạy ngay.
- **Server ở nước ngoài** (mặc định vùng Mỹ): độ trễ mạng từ Việt Nam khoảng 200 đến 300 ms. Chọn
  vùng khác tốn thêm 1,15 đến 1,75 lần giá.
- **URL public**: bắt buộc đặt `VIENEU_API_KEY` khi deploy, vì ai có URL cũng gọi được.

## 2. Cài đặt một lần

```bash
pip install modal          # hoặc: uv pip install --python .venv/Scripts/python.exe modal
modal setup                # mở trình duyệt, đăng nhập, tạo token
```

## 3. Deploy

Chạy từ thư mục gốc repo (nơi có `src/`, `apps/`, `deploy/`):

```powershell
$env:VIENEU_API_KEY = "chuoi-bi-mat-cua-ban"
modal deploy deploy/modal_app.py
```

Lần đầu Modal build image (cài gói Python, tải khoảng 300 MB model) mất vài phút; các lần sau
chỉ build lại phần thay đổi. Kết thúc, terminal in ra URL:

```
https://<workspace>--vieneu-tts.modal.run
```

Giọng người dùng trên máy deploy (`~/.cache/vieneu/user_voices.json`, đang có `clone_A`) được nhúng
vào image và chép vào Volume `vieneu-voices` ở lần chạy đầu. Giọng thêm sau đó qua `POST /voices`
cũng ghi vào Volume (Modal tự commit định kỳ và khi container tắt; file deploy còn commit ngay sau
mỗi lần thêm/xoá giọng) nên không mất khi container tắt.

Thử:

```bash
curl https://<workspace>--vieneu-tts.modal.run/health
curl -X POST https://<workspace>--vieneu-tts.modal.run/tts \
  -H "X-API-Key: chuoi-bi-mat-cua-ban" -H "Content-Type: application/json" \
  -d '{"text": "Xin chào, đây là giọng clone A chạy trên Modal.", "voice": "clone_A"}' -o out.wav
```

Trong app của bạn chỉ cần đổi `http://<HOST>:8000` trong [PROMPT_TICH_HOP_APP.md](PROMPT_TICH_HOP_APP.md)
thành URL này và gửi header `X-API-Key`.

## 4. Tuỳ chỉnh

Biến môi trường đọc lúc chạy `modal deploy`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODAL_CPU` | `2` | số **nhân vật lý** (Modal: 1 nhân = 2 vCPU). `1` rẻ hơn nhưng chậm hơn realtime với câu dài; `4` cho văn bản dài |
| `MODAL_MEMORY_MB` | `4096` | RAM (MiB); model dùng khoảng 2,3 GB đỉnh. Modal tính tiền theo mức cao hơn giữa yêu cầu và thực dùng |
| `MODAL_SCALEDOWN_SECONDS` | `120` | giữ container ấm bao lâu sau request cuối (2 giây đến 20 phút). Lúc ấm vẫn tính tiền theo cấu hình đã yêu cầu |
| `MODAL_MEMORY_SNAPSHOT` | `0` | đặt `1` để thử memory snapshot (khởi động lạnh nhanh hơn). Tài liệu Modal không nói gì về ONNX Runtime, cần thử thực tế; lỗi thì tắt |
| `VIENEU_API_KEY` | trống | API key gửi cho client; **nên đặt** |

Lệnh hữu ích:

```bash
modal app list                         # các app đã deploy
modal app logs vieneu-tts              # log realtime (thấy cold start, request)
modal app stop vieneu-tts              # tắt hẳn (không tốn credit)
modal volume ls vieneu-voices          # xem file giọng trên Volume
modal volume get vieneu-voices user_voices.json ./backup_user_voices.json
modal serve deploy/modal_app.py        # chạy thử tạm thời (URL có hậu tố -dev), tự reload khi sửa code
```

Muốn quản lý API key bằng Modal Secrets thay vì biến môi trường máy deploy:

```bash
modal secret create vieneu-api VIENEU_API_KEY=chuoi-bi-mat
```

rồi trong `deploy/modal_app.py` thay `modal.Secret.from_dict(...)` bằng `modal.Secret.from_name("vieneu-api")`.

## 5. Chi phí

Giá Functions trên trang giá Modal (không tính GPU, không chọn vùng):

| Tài nguyên | Giá |
|---|---|
| CPU (1 nhân vật lý = 2 vCPU) | 0,0000131 USD/giây, tức 0,047 USD/giờ; tối thiểu 0,125 nhân mỗi container |
| RAM | 0,00000222 USD/GiB/giây, tức 0,008 USD/GiB/giờ |
| Volume | 0,09 USD/GiB/tháng, miễn phí 1 TiB/tháng |
| GPU T4 (nếu muốn) | 0,000164 USD/giây, tức 0,59 USD/giờ |

Quy ra cấu hình:

| Cấu hình | Giá mỗi giờ chạy | 30 USD được |
|---|---|---|
| 1 nhân (2 vCPU) + 4 GiB | 0,079 USD | khoảng 380 giờ |
| 2 nhân (4 vCPU) + 4 GiB, mặc định | 0,126 USD | khoảng 237 giờ |
| 4 nhân (8 vCPU) + 8 GiB | 0,253 USD | khoảng 119 giờ |

Cách tính: Modal tính tiền theo giây khi container đang sống, theo mức cao hơn giữa tài nguyên
yêu cầu và thực dùng, kể cả lúc container ấm chờ request. Không có phí tối thiểu theo phút. Vì
thế nếu app gọi rải rác cả ngày, chi phí chủ yếu là khởi động lạnh và thời gian giữ ấm; gọi dồn
theo đợt thì hầu như chỉ tốn thời gian sinh giọng.

Ví dụ: 1.000 request/tháng, mỗi request 5 giây, 300 lần khởi động lạnh 15 giây, giữ ấm 120 giây
sau mỗi đợt (300 đợt) ở cấu hình mặc định: (5.000 + 4.500 + 36.000) giây ≈ 12,6 giờ ≈ 1,6 USD.

## 6. Giới hạn cần nhớ

- **Web request tối đa 150 giây** trên `*.modal.run`. Ở 2 nhân vật lý, tốc độ sinh khoảng 1 đến 1,5 lần
  realtime, nên một request nên dưới khoảng 1.000 đến 1.500 ký tự (60 đến 90 giây audio). Văn bản dài
  hơn thì cắt ở phía client, hoặc dùng `POST /tts/stream` (audio bắt đầu về sau chưa đến 1 giây).
- Gói Starter ghi "Web Functions (limited)", tối đa 100 container chạy đồng thời, log giữ 1 ngày.
- Một container xử lý tuần tự từng request sinh giọng (hàng chờ 16, vượt thì `429`). Modal có thể
  tự mở thêm container khi quá tải, mỗi container nạp model riêng và tính tiền riêng.
- Volume: nhiều container ghi cùng lúc thì bản ghi sau thắng; thêm giọng nên làm từ một chỗ.
  Container mới nạp file giọng lúc khởi động, container đang chạy không tự thấy giọng do container
  khác thêm.
- Gọi thẳng từ trình duyệt (CORS) phải xong trong 150 giây vì cơ chế chuyển hướng chờ của Modal
  không gắn được header CORS.
