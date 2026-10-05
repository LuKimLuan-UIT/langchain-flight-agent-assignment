"""
agents/plan_execute_agent.py — Mẫu thiết kế Plan-then-Execute
═══════════════════════════════════════════════════════════════
Plan-then-Execute = model lên kế hoạch TOÀN BỘ trước, rồi code chạy từng bước.

Quy trình:
    ┌──────────────────────────────────────────────────────────┐
    │  1. SEARCH: Harness tự gọi search_flight (từ data)      │
    │       ↓                                                  │
    │  2. PLAN: Model nhận kết quả search + constraints        │
    │          → viết TOÀN BỘ kế hoạch (1 LLM call)           │
    │       ↓                                                  │
    │  3. REVIEW: In kế hoạch → [tùy chọn] human approve      │
    │       ↓                                                  │
    │  4. EXECUTE: Code chạy từng step trong plan              │
    │          → mỗi step qua harness.check_permission()       │
    │          → nếu 1 step fail → DỪNG (không thể sửa plan)  │
    │       ↓                                                  │
    │  5. VERIFY: Harness kiểm is_done()                       │
    └──────────────────────────────────────────────────────────┘

ƯU ĐIỂM:
    + Kế hoạch rõ ràng, có thể review trước khi chạy
    + Ít LLM call (thường 1-2 call)
    + Dễ debug vì plan là structured data

NHƯỢC ĐIỂM:
    - Không linh hoạt: nếu 1 bước fail, plan không tự sửa
    - Cần model mạnh để lên plan chính xác ngay lần đầu
    - Phải có dữ liệu trước (search results) mới plan được
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .base import BaseFlightAgent
from lib.tool_call import search_flight, TOOL_MAP


# ═════════════════════════════════════════════════════════════════════
# PLAN SCHEMA — cấu trúc mà model phải trả về
# ═════════════════════════════════════════════════════════════════════
class PlanStep(BaseModel):
    """Một bước trong kế hoạch."""
    tool: str = Field(description="Tên tool: dat_ve, thanh_toan, xem_ve, huy_ve, hoi_khach_hang, xem_chinh_sach_hang")
    args: dict = Field(description="Arguments cho tool. Dùng '$pnr' nếu chưa biết mã PNR.")
    ly_do: str = Field(description="Tại sao thực hiện bước này", default="")


class Plan(BaseModel):
    """Kế hoạch đặt vé. Danh sách rỗng = không có chuyến phù hợp."""
    steps: list[PlanStep] = Field(default_factory=list)
    ghi_chu: str = Field(description="Ghi chú tổng quan", default="")


PLANNER_PROMPT = """Bạn là planner cho hệ thống đặt vé máy bay.

NHIỆM VỤ: Viết kế hoạch đặt vé dưới dạng danh sách các bước.

CÁC TOOL CÓ SẴN (không bao gồm search_flight, vì đã tìm sẵn):
- dat_ve(ma_chuyen_bay, so_ve, ten_khach): Đặt vé, trả về mã PNR
- thanh_toan(pnr): Thanh toán cho booking. Dùng "$pnr" nếu chưa biết mã.
- xem_ve(pnr): Xem lại booking. Dùng "$pnr" nếu chưa biết mã.
- huy_ve(pnr): Hủy vé chưa thanh toán.
- hoi_khach_hang(cau_hoi, lua_chon): Hỏi khách khi không có vé phù hợp.
- xem_chinh_sach_hang(hang): Tra cứu chính sách hãng.

QUY TẮC:
1. Xem danh sách chuyến bay đã tìm sẵn
2. Chọn chuyến bay thỏa MỌI ràng buộc (chặng, ngày, giờ, giá, còn chỗ)
3. Viết plan: dat_ve → thanh_toan → xem_ve
4. Nếu KHÔNG có chuyến nào thỏa mãn, có thể thêm bước hoi_khach_hang hoặc trả về danh sách rỗng + ghi chú lý do

QUAN TRỌNG: Trả về JSON theo schema Plan. Nếu không có chuyến phù hợp, 
trả steps=[] và ghi_chu giải thích."""


class PlanExecuteFlightAgent(BaseFlightAgent):
    """Agent đặt vé máy bay theo mẫu Plan-then-Execute.

    Model gọi 1 lần để tạo toàn bộ kế hoạch.
    Code chạy từng bước, mỗi bước qua harness.
    Nếu 1 bước fail → dừng (plan không thể tự sửa).
    """

    pattern_name = "Plan-then-Execute"

    def __init__(self, *args, max_plans: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_plans = max_plans   # tối đa bao nhiêu lần viết plan

    def run(self, user_request: str) -> dict:
        self.reset()
        self.start_time = time.time()
        self._final_answer = ""

        c = self.constraints

        # ─── BƯỚC 1: SEARCH (harness tự gọi, dùng data từ constraints) ───
        self._log("🔍 Bước 1: Tìm chuyến bay (từ constraints data)...")
        search_result = self._execute_tool_with_harness(
            "search_flight",
            {"tu": c.tu, "den": c.den, "ngay": c.ngay}
        )

        if search_result.get("status") != "ok":
            self._log(f"❌ Không tìm thấy chuyến bay: {search_result}")
            self._final_answer = "Không tìm thấy chuyến bay phù hợp."
            return self._build_result()

        flights_json = json.dumps(search_result["ket_qua"], ensure_ascii=False, indent=2)

        # ─── BƯỚC 2: PLAN (model viết kế hoạch) ───
        plan = None
        for attempt in range(1, self.max_plans + 1):
            self._log(f"📋 Bước 2: Lên kế hoạch (lần {attempt}/{self.max_plans})...")

            planner_llm = self.llm.with_structured_output(Plan)
            self.total_llm_calls += 1

            prompt_text = (
                f"{PLANNER_PROMPT}\n\n"
                f"YÊU CẦU CỦA KHÁCH: {user_request}\n\n"
                f"RÀNG BUỘC:\n{c.to_prompt()}\n\n"
                f"CHUYẾN BAY ĐÃ TÌM ĐƯỢC:\n{flights_json}"
            )

            try:
                draft: Plan = planner_llm.invoke(prompt_text)
            except Exception as e:
                self._log(f"❌ LLM error khi lên plan: {e}")
                continue

            # In plan
            if not draft.steps:
                self._log(f"📋 Plan rỗng: {draft.ghi_chu}")
                self._final_answer = f"Không có chuyến bay phù hợp. {draft.ghi_chu}"
                break

            self._log(f"📋 Plan ({len(draft.steps)} bước):")
            for i, step in enumerate(draft.steps, 1):
                self._log(f"   {i}. {step.tool}({step.args}) — {step.ly_do}")
            if draft.ghi_chu:
                self._log(f"   Ghi chú: {draft.ghi_chu}")

            plan = draft
            break   # dùng plan đầu tiên (có thể thêm human review ở đây)

        if not plan or not plan.steps:
            return self._build_result()

        # ─── BƯỚC 3: EXECUTE (code chạy từng step) ───
        self._log("⚙️  Bước 3: Thực thi kế hoạch...")
        last_pnr = ""   # để fill "$pnr"

        for i, step in enumerate(plan.steps, 1):
            # Thay thế placeholder "$pnr"
            args = {}
            for k, v in step.args.items():
                if isinstance(v, str) and v.strip().lower() in ("$pnr", "$booking_code"):
                    args[k] = last_pnr
                else:
                    args[k] = v

            result = self._execute_tool_with_harness(step.tool, args)

            # Lưu PNR nếu có
            if result.get("status") == "ok" and "pnr" in result:
                last_pnr = result["pnr"]

            # Nếu bước fail → dừng (plan-then-execute không thể sửa plan)
            if result.get("status") not in ("ok",):
                self._log(f"❌ Step {i} thất bại. Plan-then-Execute không thể sửa plan → dừng.")
                self._final_answer = (
                    f"Kế hoạch thất bại ở bước {i}: {step.tool}({args}). "
                    f"Lý do: {result.get('reason', result.get('error', result.get('hint', '?')))}"
                )
                break

        # ─── BƯỚC 4: VERIFY ───
        if self.harness.is_done():
            self._log("✅ Harness xác nhận: ĐÃ HOÀN THÀNH!")
            booking = self.harness.get_completed_booking()
            if booking:
                self._final_answer = (
                    f"Đặt vé thành công! PNR: {booking['pnr']}, "
                    f"Chuyến: {booking['ma_chuyen_bay']}, "
                    f"Giá: {booking['gia_moi_ve']:,} VND, "
                    f"Ngày: {booking['ngay_bay']} {booking['gio_bay']}"
                )

        return self._build_result()
