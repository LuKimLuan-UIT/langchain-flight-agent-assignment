"""
app.py — Giao diện Web tương tác (Streamlit UI/UX) cho Flight Booking Agent
════════════════════════════════════════════════════════════════════════════
Cho phép:
  1. Trải nghiệm trực quan 3 mẫu thiết kế Agent (ReAct, Plan-then-Execute, Lai).
  2. Kết nối trực tiếp với Google Gemini API (gemini-2.0-flash, gemini-1.5-flash, ...).
  3. Theo dõi live trace: các bước suy luận, lệnh gọi tool, harness permission check.
  4. Xem kết quả đặt vé hoặc gói bàn giao (Handoff) cho con người.
  5. Chạy benchmark so sánh hiệu quả giữa 3 mẫu thiết kế.
  6. Tra cứu & quản lý cơ sở dữ liệu mockup chuyến bay.

Chạy:
  streamlit run app.py
"""

import json
import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load môi trường
load_dotenv()

from lib.tool_call import CHUYEN_BAY, DA_DAT, reset_db
from lib.harness import RangBuoc
from agents.react_agent import ReactFlightAgent
from agents.plan_execute_agent import PlanExecuteFlightAgent
from agents.hybrid_agent import HybridFlightAgent
from evaluate import TEST_CASES

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Flight Booking Agent - SE373 BTVN#3",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .trace-card {
        background-color: #0F172A;
        color: #F1F5F9;
        font-family: monospace;
        font-size: 0.88rem;
        padding: 14px;
        border-radius: 6px;
        overflow-x: auto;
    }
    .success-box {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 6px solid #10B981;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .success-box h4 {
        color: #10B981 !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.6rem;
    }
    .success-box p {
        margin-bottom: 0.4rem;
        line-height: 1.6;
    }
    .success-box code {
        font-size: 1.05rem;
        font-weight: 700;
        color: #10B981 !important;
        background-color: rgba(16, 185, 129, 0.18) !important;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .fail-box {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 6px solid #EF4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .fail-box h4 {
        color: #EF4444 !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.6rem;
    }
    .fail-box p, .fail-box li {
        margin-bottom: 0.4rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR — Cấu hình API & Model
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2200/2200326.png", width=64)
    st.markdown("### ⚙️ Cấu hình hệ thống")
    
    # 0. Chế độ No-API
    no_api_mode = st.toggle(
        "⚡ Chế độ NO-API (Giả lập)",
        value=not bool(os.getenv("GOOGLE_API_KEY")),
        help="Chạy bằng Model giả lập (ModelGiaFlight): Không cần API Key, không lo Rate Limit (429), tốc độ tức thì!",
    )

    # 1. API Key (nếu không dùng No-API)
    default_api_key = os.getenv("GOOGLE_API_KEY", "")
    if not no_api_mode:
        api_key = st.text_input(
            "Google Gemini API Key",
            value=default_api_key,
            type="password",
            help="Lấy API Key tại: https://aistudio.google.com/apikey",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.info("🟢 **Đang bật NO-API:** Hoàn toàn offline, không cần API Key!")

    # 2. Chọn Model
    if no_api_mode:
        model_options = ["[MOCK] Model Giả Lập (Không cần API Key)"]
        model_name = "mock"
        st.selectbox("Mô hình LLM", options=model_options, index=0)
        
        kich_ban_mock = st.selectbox(
            "Kịch bản giả lập:",
            options=[
                "tu_dong (Tự động theo yêu cầu)",
                "lap (Kiểm thử LoopDetector)",
                "vi_pham_quyen (Kiểm thử Permission Check)",
                "bip_grounding (Kiểm thử Grounding Check)",
            ],
            index=0,
            help="Chọn kịch bản đặc biệt để kiểm thử các tầng của Harness",
        )
        kich_ban_code = kich_ban_mock.split(" ")[0]
    else:
        model_options = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash", "[MOCK] Model Giả Lập"]
        selected_model = st.selectbox("Mô hình LLM (Google)", options=model_options, index=0)
        model_name = "mock" if "[MOCK]" in selected_model else selected_model
        kich_ban_code = "tu_dong"

    # 3. Chọn Mẫu thiết kế Agent
    st.markdown("---")
    st.markdown("### 🤖 Mẫu thiết kế Agent")
    pattern_choice = st.radio(
        "Kiến trúc:",
        options=["ReAct", "Plan-then-Execute", "Hybrid (Lai)", "Mock-Agent (Độc lập)"],
        index=0,
        help="Chọn mô hình kiến trúc Agent để thử nghiệm",
    )

    st.markdown("---")
    st.caption("SE373 — Kỹ thuật phần mềm ứng dụng AI")
    st.caption("Sinh viên: Lư Kim Luân · MSSV: 24521028")


# Header
st.markdown('<div class="main-title">🛫 Flight Booking Agent with Safety Harness</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống đặt vé máy bay thông minh với 4 tầng bảo vệ Harness và 3 mẫu thiết kế Agent (SE373 BTVN#3)</div>', unsafe_allow_html=True)

# Tabs chức năng
tab_booking, tab_eval, tab_db, tab_docs = st.tabs([
    "✈️ Đặt vé & Thử nghiệm", 
    "📊 Benchmark 3 Mẫu thiết kế", 
    "🗄️ Tra cứu chuyến bay Mockup",
    "📖 Hướng dẫn & Kiến trúc"
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: ĐẶT VÉ & THỬ NGHIỆM
# ═════════════════════════════════════════════════════════════════════
with tab_booking:
    col_preset, col_reset = st.columns([4, 1])
    with col_preset:
        preset = st.selectbox(
            "⚡ Chọn kịch bản mẫu nhanh:",
            options=[
                "Tùy chỉnh tự do",
                "Kịch bản 1 (EASY): SGN → HAN, sáng sớm, max 2.000.000đ (Thành công)",
                "Kịch bản 2 (NO_FIT): SGN → HAN, sáng sớm, max 1.000.000đ (Giá quá thấp, Bàn giao)",
                "Kịch bản 3 (SOLD_OUT): SGN → CXR, chuyến duy nhất hết chỗ (Bàn giao)",
            ]
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 Reset DB"):
            reset_db()
            st.toast("Đã khôi phục dữ liệu ban đầu!", icon="✅")

    # Điền giá trị theo preset
    val_tu, val_den, val_ngay, val_gio, val_gia, val_ve, val_ten = "SGN", "HAN", "2026-10-01", "23:59", 2_000_000, 1, "Nguyen Van A"
    if "Kịch bản 1" in preset:
        val_tu, val_den, val_ngay, val_gio, val_gia = "SGN", "HAN", "2026-10-01", "07:00", 2_000_000
    elif "Kịch bản 2" in preset:
        val_tu, val_den, val_ngay, val_gio, val_gia = "SGN", "HAN", "2026-10-01", "10:00", 1_000_000
    elif "Kịch bản 3" in preset:
        val_tu, val_den, val_ngay, val_gio, val_gia = "SGN", "CXR", "2026-10-01", "23:59", 2_000_000

    with st.form("booking_form"):
        st.markdown("##### 📋 Thông tin & Ràng buộc đặt vé (Constraints as DATA)")
        c1, c2, c3 = st.columns(3)
        with c1:
            input_tu = st.text_input("Điểm đi (Mã IATA)", value=val_tu).upper()
            input_den = st.text_input("Điểm đến (Mã IATA)", value=val_den).upper()
        with c2:
            input_ngay = st.text_input("Ngày bay (YYYY-MM-DD)", value=val_ngay)
            input_gio = st.text_input("Giờ bay muộn nhất (HH:MM)", value=val_gio)
        with c3:
            input_gia = st.number_input("Giá tối đa (VND)", value=val_gia, step=100_000)
            input_ve = st.number_input("Số lượng vé", value=val_ve, min_value=1, max_value=10)
        input_ten = st.text_input("Họ và tên hành khách", value=val_ten)

        submit = st.form_submit_button("🚀 Chạy Agent Đặt Vé", use_container_width=True)

    if submit:
        is_mock_run = no_api_mode or model_name == "mock"
        if not is_mock_run and not os.getenv("GOOGLE_API_KEY"):
            st.error("⚠️ Vui lòng nhập Google Gemini API Key tại thanh bên trái trước khi chạy (hoặc bật chế độ NO-API)!")
        else:
            constraints = RangBuoc(
                tu=input_tu,
                den=input_den,
                ngay=input_ngay,
                gio_muon_nhat=input_gio,
                gia_toi_da=input_gia,
                so_ve=input_ve,
                ten_khach=input_ten,
            )

            # Chọn class Agent
            from agents.base import MockFlightAgent
            agent_map = {
                "ReAct": ReactFlightAgent,
                "Plan-then-Execute": PlanExecuteFlightAgent,
                "Hybrid (Lai)": HybridFlightAgent,
                "Mock-Agent (Độc lập)": MockFlightAgent,
            }
            AgentClass = agent_map[pattern_choice]

            with st.spinner(f"Agent đang xử lý theo mô hình {pattern_choice}..."):
                agent = AgentClass(
                    constraints=constraints,
                    model_name=model_name,
                    verbose=False,
                    kich_ban=kich_ban_code if is_mock_run else "tu_dong",
                )
                
                try:
                    result = agent.run(constraints.to_prompt())
                except Exception as e:
                    st.error(f"Lỗi khi thực thi: {e}")
                    result = None

            if result:
                ket_qua = result.get("ket_qua")
                metrics = result.get("metrics", {})

                # Hiển thị số liệu hiệu năng
                st.markdown("##### 📈 Chỉ số hiệu năng")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mẫu thiết kế", metrics.get("pattern", pattern_choice))
                m2.metric("Số lượt gọi LLM", metrics.get("llm_calls", 0))
                m3.metric("Số lượt gọi Tool", metrics.get("tool_calls", 0))
                m4.metric("Thời gian thực thi", f"{metrics.get('time_seconds', 0)}s")

                st.markdown("---")

                # Kết quả chi tiết
                if ket_qua == "THANH_CONG":
                    booking = result.get("booking", {})
                    ten_khach_hien_thi = booking.get("ten_khach", "")
                    if "cho " in ten_khach_hien_thi.lower():
                        ten_khach_hien_thi = ten_khach_hien_thi.split("cho ")[-1].strip()
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>🎉 ĐẶT VÉ THÀNH CÔNG (Harness xác nhận)</h4>
                        <p><b>Mã PNR:</b> <code>{booking.get('pnr')}</code></p>
                        <p><b>Chuyến bay:</b> {booking.get('ma_chuyen_bay')} ({booking.get('hang')})</p>
                        <p><b>Hành trình:</b> {booking.get('tu')} ➔ {booking.get('den')} | <b>Khởi hành:</b> {booking.get('gio_bay')} ngày {booking.get('ngay_bay')}</p>
                        <p><b>Hành khách:</b> {ten_khach_hien_thi} ({booking.get('so_ve')} vé)</p>
                        <p><b>Tổng tiền:</b> {booking.get('tong_tien', 0):,} VND | <b>Trạng thái:</b> ĐÃ THANH TOÁN ✅</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    ban_giao = result.get("ban_giao", {})
                    st.markdown(f"""
                    <div class="fail-box">
                        <h4>⚠️ KHÔNG THỂ HOÀN THÀNH — BÀN GIAO CHO CON NGƯỜI (HANDOFF)</h4>
                        <p><b>❓ Câu hỏi cần quyết định:</b><br>{ban_giao.get('cau_hoi')}</p>
                        <p><b>📌 Công việc đã làm:</b> {", ".join(ban_giao.get('da_lam', []))}</p>
                        <p><b>🛠️ Các bước đã thử:</b></p>
                        <ul>{"".join(f"<li>{item}</li>" for item in ban_giao.get('da_thu', []))}</ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Hiển thị Trace thực thi
                with st.expander("🔍 Xem chi tiết vết thực thi (Execution Trace & Harness Log)", expanded=True):
                    trace_text = result.get("trace", "Không có trace")
                    st.markdown(f'<div class="trace-card">{trace_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                # Kiểm tra Grounding
                if "grounding" in result:
                    with st.expander("🛡️ Kiểm tra căn cứ dữ liệu (Grounding Check)"):
                        gd = result["grounding"]
                        if gd.get("dat"):
                            st.success("✅ Mọi dữ kiện trong câu trả lời đều có nguồn gốc từ kết quả Tool.")
                        else:
                            st.warning(f"⚠️ Phát hiện dữ kiện không rõ nguồn: {gd.get('khong_co_nguon', [])}")
                        st.json(gd.get("chi_tiet", gd))


# ═════════════════════════════════════════════════════════════════════
# TAB 2: BENCHMARK 3 MẪU THIẾT KẾ
# ═════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 Đánh giá so sánh 3 Mẫu thiết kế Agent")
    st.write("Chạy cùng lúc 3 kịch bản chuẩn (EASY, NO_FIT, SOLD_OUT) qua cả 3 mẫu kiến trúc để đo lường độ chính xác và mức độ tiêu hao tài nguyên.")
    if no_api_mode or model_name == "mock":
        st.info("⚡ **Chế độ Benchmark NO-API:** Sử dụng model giả lập để đánh giá tức thì, tránh 429 Rate Limit!")

    if st.button("🚀 Bắt đầu Benchmark Đánh Giá Toàn Diện", type="primary"):
        is_mock_eval = no_api_mode or model_name == "mock"
        if not is_mock_eval and not os.getenv("GOOGLE_API_KEY"):
            st.error("⚠️ Vui lòng cung cấp Google Gemini API Key tại thanh bên trái (hoặc bật chế độ NO-API)!")
        else:
            eval_model = "mock" if is_mock_eval else model_name
            agent_classes = [
                ("ReAct", ReactFlightAgent),
                ("Plan-then-Execute", PlanExecuteFlightAgent),
                ("Hybrid", HybridFlightAgent),
            ]

            eval_results = []
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            total_steps = len(TEST_CASES) * len(agent_classes)
            current_step = 0

            for tc in TEST_CASES:
                for pattern_name, AgentClass in agent_classes:
                    current_step += 1
                    status_text.text(f"Đang chạy {pattern_name} trên kịch bản: {tc['ten']}...")
                    progress_bar.progress(current_step / total_steps)

                    constraints = tc["constraints"]
                    agent = AgentClass(
                        constraints=constraints,
                        model_name=eval_model,
                        verbose=False,
                    )
                    
                    try:
                        res = agent.run(constraints.to_prompt())
                        ket_qua_thuc = res.get("ket_qua")
                        metrics = res.get("metrics", {})
                        dung = (ket_qua_thuc == tc["ket_qua_mong_doi"])
                        
                        eval_results.append({
                            "Kịch bản": tc["ten"].split(" — ")[0],
                            "Pattern": pattern_name,
                            "Kết quả": ket_qua_thuc,
                            "Chính xác": "✅ Đạt" if dung else "❌ Lỗi",
                            "LLM Calls": metrics.get("llm_calls", 0),
                            "Tool Calls": metrics.get("tool_calls", 0),
                            "Thời gian (s)": metrics.get("time_seconds", 0),
                        })
                    except Exception as e:
                        eval_results.append({
                            "Kịch bản": tc["ten"].split(" — ")[0],
                            "Pattern": pattern_name,
                            "Kết quả": "LOI",
                            "Chính xác": "❌ Lỗi",
                            "LLM Calls": agent.total_llm_calls,
                            "Tool Calls": agent.total_tool_calls,
                            "Thời gian (s)": 0,
                        })

            status_text.text("✅ Hoàn thành Benchmark!")
            progress_bar.empty()

            st.markdown("#### 📋 Bảng tổng hợp kết quả Benchmark")
            st.dataframe(eval_results, use_container_width=True)

            # Thống kê trung bình theo từng Pattern
            st.markdown("#### 📈 So sánh chỉ số trung bình theo Mẫu thiết kế")
            c_re, c_pl, c_hy = st.columns(3)
            
            for col, p_name in [(c_re, "ReAct"), (c_pl, "Plan-then-Execute"), (c_hy, "Hybrid")]:
                p_rows = [r for r in eval_results if r["Pattern"] == p_name]
                acc = sum(1 for r in p_rows if "Đạt" in r["Chính xác"]) / len(p_rows) * 100
                avg_llm = sum(r["LLM Calls"] for r in p_rows) / len(p_rows)
                avg_time = sum(r["Thời gian (s)"] for r in p_rows) / len(p_rows)

                with col:
                    st.markdown(f"##### {p_name}")
                    st.metric("Độ chính xác", f"{acc:.0f}%")
                    st.metric("Trung bình LLM Calls", f"{avg_llm:.1f}")
                    st.metric("Thời gian TB", f"{avg_time:.1f}s")


# ═════════════════════════════════════════════════════════════════════
# TAB 3: TRA CỨU CHUYẾN BAY MOCKUP
# ═════════════════════════════════════════════════════════════════════
with tab_db:
    st.markdown("### 🗄️ Dữ liệu chuyến bay nội địa (Mockup Database)")
    st.write("Bảng dữ liệu chuyến bay mẫu trong bộ nhớ được Agent sử dụng.")
    
    st.dataframe(CHUYEN_BAY, use_container_width=True)

    st.markdown("### 📑 Các vé hiện đang được đặt trong hệ thống (`DA_DAT`)")
    if DA_DAT:
        st.json(DA_DAT)
    else:
        st.info("Hiện tại chưa có vé nào được đặt trong hệ thống.")


# ═════════════════════════════════════════════════════════════════════
# TAB 4: HƯỚNG DẪN & KIẾN TRÚC
# ═════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("""
    ### 🛡️ 4 Trụ cột của Bộ dây đai an toàn (Harness)
    1. **Ràng buộc là DỮ LIỆU (Constraints as DATA):**
       Yêu cầu của khách được lưu trong dataclass `RangBuoc`. Code Python kiểm tra bằng logic toán học, LLM không thể tự ý "quên" ràng buộc.
    2. **Kiểm quyền TRƯỚC khi gọi Tool (Permission Check):**
       Mỗi khi LLM muốn gọi `dat_ve` hay `thanh_toan`, Harness kiểm tra điều kiện trước. Nếu vi phạm, cuộc gọi bị chặn ngay lập tức.
    3. **Tiêu chí hoàn thành kiểm bằng CODE (Done by Code):**
       Không bao giờ tin vào lời khẳng định của LLM. Hàm `is_done()` đọc lại dữ liệu thực tế từ database (`DA_DAT`) để xác nhận vé đã thanh toán hợp lệ.
    4. **Bàn giao có cấu trúc (Handoff to Human):**
       Khi thất bại, Agent cung cấp 3 trường thông tin: việc đã làm, việc đã thử, và câu hỏi cụ thể xin ý kiến con người.

    ---
    ### 🤖 So sánh 3 Mẫu thiết kế Agent
    - **ReAct:** Từng bước một (Reason $\\rightarrow$ Act $\\rightarrow$ Observe). Rất linh hoạt, thích ứng tốt, nhưng tốn nhiều LLM call.
    - **Plan-then-Execute:** LLM lập kế hoạch 1 lần duy nhất, sau đó code chạy tuần tự. Nhanh và tiết kiệm, nhưng không thể tự sửa nếu 1 bước bị gãy.
    - **Lai (Hybrid):** Chạy Plan trước để tối ưu chi phí. Nếu gặp lỗi, lập tức kích hoạt ReAct để tự động tìm giải pháp cứu hộ.
    """)
