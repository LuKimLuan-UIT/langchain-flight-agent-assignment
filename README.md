# 🛫 HƯỚNG DẪN SỬ DỤNG — BTVN#3: FLIGHT BOOKING AGENT WITH SAFETY HARNESS

---

## 📌 Giới thiệu tổng quan

Dự án hiện thực hóa hệ thống **Agent AI đặt vé máy bay tự động** tích hợp **Khung an toàn 4 trụ cột (Safety Harness)** để kiểm soát rủi ro, chống ảo giác và ngăn chặn vòng lặp vô tận.

Hệ thống so sánh và đánh giá **3 mẫu thiết kế Agent (Agent Architecture Patterns)**:
1. **ReAct Agent:** Suy luận và gọi công cụ từng bước (Reasoning + Acting), linh hoạt thích ứng khi có sự cố.
2. **Plan-then-Execute Agent:** Lập kế hoạch toàn bộ trước (Structured Output), thực thi tuần tự theo danh sách bước đã định.
3. **Hybrid Agent (Lai):** Chạy Plan-then-Execute trước để tối ưu chi phí LLM, nếu gặp lỗi giữa chừng sẽ tự động kích hoạt ReAct Fallback để cứu hộ.
4. *(Bổ trợ)* **Mock Agent / Model Giả Lập:** Chế độ không cần API Key, giả lập chính xác hành vi LLM phục vụ kiểm thử nhanh và tránh lỗi giới hạn lượt gọi (Rate Limit 429).

---

## 📂 Cấu trúc thư mục

```text
BTVN3/
├── app.py                      # Giao diện Web UI/UX đa năng bằng Streamlit
├── main.py                     # Entry point chạy dòng lệnh (CLI interactive & arg flags)
├── evaluate.py                 # Kịch bản Benchmark & Đánh giá so sánh 3 pattern
├── requirements.txt            # Danh sách thư viện cần thiết
├── .env.example                # File mẫu cấu hình biến môi trường
├── .env                        # File cấu hình API Key (tạo từ .env.example)
├── agents/                     # Các mẫu thiết kế Agent
│   ├── base.py                 # Lớp nền tảng kết nối LLM, harness & MockFlightAgent
│   ├── react_agent.py          # Pattern ReAct
│   ├── plan_execute_agent.py   # Pattern Plan-then-Execute
│   └── hybrid_agent.py         # Pattern Hybrid (Lai)
├── lib/                        # Khung an toàn và công cụ
│   ├── harness.py              # 4 trụ cột Harness, Middleware, LoopDetector, kiem_can_cu
│   ├── tool_call.py            # 5 tool cốt lõi, 3 tool hỗ trợ & Database mockup
│   └── model_gia.py            # Bộ não giả lập LLM cho chế độ No-API
```

---

## ⚙️ Cài đặt môi trường

### 1. Chuẩn bị Python
Khuyến nghị sử dụng **Python 3.10 trở lên**.

### 2. Cài đặt các thư viện cần thiết
Mở terminal/powershell tại thư mục `BTVN3` và chạy:

```bash
pip install -r requirements.txt
```

*Các thư viện chính bao gồm: `langchain`, `langchain-google-genai`, `langgraph`, `pydantic`, `tabulate`, `python-dotenv`, `streamlit`.*

### 3. Cấu hình biến môi trường (`.env`)
Tạo file `.env` bằng cách sao chép từ `.env.example`:

```bash
cp .env.example .env
```
*(Trên Windows PowerShell: `Copy-Item .env.example .env`)*

Nội dung file `.env`:
```env
GOOGLE_API_KEY=your-actual-api-key...
DEFAULT_MODEL=your default model
```

> 💡 **Mẹo:** Nếu chưa có API Key hoặc bị chạm ngưỡng Rate Limit của Google (mã lỗi 429), bạn hoàn toàn có thể chạy chế độ **NO-API** (sử dụng Model giả lập `model_gia.py`) mà không cần cấu hình API Key!

---

## 🚀 Hướng dẫn sử dụng chi tiết

### Cách 1: Chạy giao diện Web trực quan (Streamlit) — *Khuyến nghị*

Khởi chạy ứng dụng Web với 4 Tab tính năng:
```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở trang web (mặc định tại `http://localhost:8501`). Tại đây bạn có thể:
- **Tab 1: ✈️ Đặt vé & Thử nghiệm:** Chọn kịch bản mẫu nhanh (EASY, NO_FIT, SOLD_OUT) hoặc nhập ràng buộc tùy ý. Theo dõi Trace log thời gian thực từng bước suy luận, kiểm tra quyền Harness và xem kết quả bàn giao.
- **Tab 2: 📊 Benchmark 3 Mẫu thiết kế:** Nhấn 1 nút để chạy tự động 9 lượt test và xuất bảng ma trận so sánh tỷ lệ thành công, số bước, số token/cuộc gọi LLM.
- **Tab 3: 🗄️ Tra cứu chuyến bay Mockup:** Xem và tìm kiếm cơ sở dữ liệu chuyến bay mẫu (`CHUYEN_BAY`) và danh sách vé đã đặt (`DA_DAT`), có nút Reset DB nhanh.
- **Tab 4: 📖 Hướng dẫn & Kiến trúc:** Xem sơ đồ kiến trúc và nguyên lý hoạt động 4 trụ cột Harness.

> **Lưu ý trên giao diện:** Bạn có thể bật toggle **"⚡ Chế độ NO-API (Giả lập)"** ở thanh sidebar bên trái để chạy demo tức thì mà không cần mạng hoặc API Key.

---

### Cách 2: Chạy dòng lệnh tương tác (CLI Interactive)

Chạy file `main.py` ở chế độ hỏi - đáp từng bước trên terminal:

```bash
python main.py
```

Chương trình sẽ hiển thị menu để bạn:
1. Chọn pattern Agent (`1. ReAct`, `2. Plan-then-Execute`, `3. Hybrid`, `4. Mock-Agent`).
2. Nhập các ràng buộc (Điểm đi, điểm đến, ngày bay, giờ bay, giá tối đa, số vé, tên khách). Nhấn `Enter` để chọn giá trị mặc định.
3. Xem toàn bộ log suy luận và kết quả JSON trả về.

---

### Cách 3: Chạy nhanh một Agent Pattern cụ thể

Bạn có thể truyền trực tiếp tham số `--pattern`:

```bash
# Chạy ReAct Agent
python main.py --pattern react

# Chạy Plan-then-Execute Agent
python main.py --pattern plan

# Chạy Hybrid Agent
python main.py --pattern hybrid

# Chạy Mock Agent
python main.py --pattern mock
```

---

### Cách 4: Chế độ NO-API (Chạy giả lập không cần API Key)

Khi cần chạy thử nghiệm nhanh, chấm điểm code hoặc khi mất kết nối mạng / hết hạn mức API Gemini:

```bash
# Chạy tương tác không cần API
python main.py --no-api

# Chạy pattern cụ thể không cần API
python main.py --pattern react --no-api
python main.py --pattern plan --no-api
python main.py --pattern hybrid --no-api
```

---

### Cách 5: Chạy Benchmark đánh giá hiệu năng (Evaluate)

Để chạy kiểm thử tự động toàn diện trên cả 3 kịch bản (`EASY`, `NO_FIT`, `SOLD_OUT`) qua 3 Agent pattern:

```bash
# Đánh giá sử dụng API thật
python evaluate.py

# Hoặc dùng thông qua main.py
python main.py --evaluate

# Đánh giá bằng bộ não giả lập (cực nhanh, không tốn quota)
python main.py --evaluate --no-api
```

Kết quả sẽ xuất ra bảng so sánh chi tiết dạng Markdown/Tabulate:
- Tỷ lệ thành công (%)
- Số lần gọi LLM (LLM Calls)
- Số lần gọi Tool (Tool Calls)
- Thời gian thực thi (giây)
- Lý do thất bại & tính chính xác của gói bàn giao Handoff

---

