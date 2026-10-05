"""
agents/hybrid_agent.py — Mẫu thiết kế Lai (Hybrid: Plan + ReAct fallback)
═══════════════════════════════════════════════════════════════════════════
Hybrid = Plan-then-Execute TRƯỚC, nếu plan fail thì chuyển sang ReAct.

Quy trình:
    ┌────────────────────────────────────────────────────────────────┐
    │  PHASE 1: Plan-then-Execute                                    │
    │  ─────────────────────────                                     │
    │  Search → Model lên plan → Execute từng step qua harness       │
    │       ↓ (tất cả OK)         ↓ (1 step fail)                    │
    │       THÀNH CÔNG             ↓                                  │
    │                              │                                  │
    │  PHASE 2: ReAct fallback     │                                  │
    │  ────────────────────────    ↓                                  │
    │  Model nhận: "Plan đã thất bại ở bước X, lỗi: Y"              │
    │  Model tự quyết định từng bước (ReAct loop)                    │
    │  Harness vẫn kiểm soát mỗi tool call                          │
    │       ↓                                                        │
    │  Harness kiểm is_done()                                        │
    └────────────────────────────────────────────────────────────────┘

ƯU ĐIỂM:
    + Kế hoạch rõ ràng ban đầu (từ Plan)
    + Linh hoạt xử lý lỗi (từ ReAct)
    + Tận dụng ưu điểm cả 2 pattern

NHƯỢC ĐIỂM:
    - Phức tạp nhất trong 3 pattern
    - Tốn nhiều LLM call hơn Plan-then-Execute (khi plan fail)
    - Cần quản lý state khi chuyển phase
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from pydantic import BaseModel, Field

from .base import BaseFlightAgent
from lib.tool_call import search_flight, TOOL_MAP


# ═════════════════════════════════════════════════════════════════════
# PLAN SCHEMA (giống Plan-then-Execute)
# ═════════════════════════════════════════════════════════════════════
class PlanStep(BaseModel):
    tool: str = Field(description="Tên tool: dat_ve, thanh_toan, xem_ve, huy_ve, hoi_khach_hang, xem_chinh_sach_hang")
    args: dict = Field(description="Arguments. Dùng '$pnr' nếu chưa biết PNR.")
    ly_do: str = Field(default="")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    ghi_chu: str = Field(default="")


PLANNER_PROMPT = """Bạn là planner cho hệ thống đặt vé máy bay.
Viết kế hoạch đặt vé dưới dạng danh sách bước.

TOOL (không bao gồm search_flight):
- dat_ve(ma_chuyen_bay, so_ve, ten_khach)
- thanh_toan(pnr) — dùng "$pnr" nếu chưa biết
- xem_ve(pnr) — dùng "$pnr"
- huy_ve(pnr)
- hoi_khach_hang(cau_hoi, lua_chon)
- xem_chinh_sach_hang(hang)

Nếu không có chuyến phù hợp, trả steps=[] + ghi_chu."""


REACT_RECOVERY_PROMPT = """Bạn là agent đặt vé máy bay. Kế hoạch trước đó đã THẤT BẠI.

TÌNH HUỐNG: {failure_context}

Bạn cần tự quyết định từng bước để giải quyết vấn đề hoặc tìm phương án thay thế.

CÁC TOOL:
1. search_flight(tu, den, ngay) — Tìm chuyến bay
2. dat_ve(ma_chuyen_bay, so_ve, ten_khach) — Đặt vé
3. thanh_toan(pnr) — Thanh toán
4. xem_ve(pnr) — Xem booking
5. huy_ve(pnr) — Hủy vé
6. goi_y_chuyen_bay_thay_the(tu, den, ngay, gia_toi_da) — Tìm chuyến bay thay thế ngày khác/giờ khác
7. hoi_khach_hang(cau_hoi, lua_chon) — Giao tiếp hỏi ý kiến khách hàng (vd: nới lỏng điều kiện)
8. xem_chinh_sach_hang(hang) — Tra cứu chính sách hãng

RÀNG BUỘC CỦA KHÁCH: {constraints}

Hãy thử tìm giải pháp khác hoặc dùng hoi_khach_hang nếu cần. Trả lời tiếng Việt."""


class HybridFlightAgent(BaseFlightAgent):
    """Agent đặt vé theo mẫu Lai: Plan trước, ReAct khi plan fail.

    Phase 1: Model lên plan → code execute → nếu OK → xong.
    Phase 2: Nếu plan fail → chuyển sang ReAct loop, model tự quyết.
    """

    pattern_name = "Hybrid"

    def __init__(self, *args, max_react_steps: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_react_steps = max_react_steps

    def run(self, user_request: str) -> dict:
        self.reset()
        self.start_time = time.time()
        self._final_answer = ""

        c = self.constraints

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: PLAN-THEN-EXECUTE
        # ═══════════════════════════════════════════════════════════
        self._log("━━━ PHASE 1: Plan-then-Execute ━━━")

        # Search
        self._log("🔍 Tìm chuyến bay...")
        search_result = self._execute_tool_with_harness(
            "search_flight",
            {"tu": c.tu, "den": c.den, "ngay": c.ngay}
        )

        plan_failed = False
        failure_context = ""

        if search_result.get("status") != "ok":
            plan_failed = True
            failure_context = f"Không tìm thấy chuyến bay {c.tu}→{c.den} ngày {c.ngay}."
        else:
            # Lên plan
            flights_json = json.dumps(search_result["ket_qua"], ensure_ascii=False, indent=2)

            self._log("📋 Model lên kế hoạch...")
            planner_llm = self.llm.with_structured_output(Plan)
            self.total_llm_calls += 1

            try:
                draft: Plan = planner_llm.invoke(
                    f"{PLANNER_PROMPT}\n\n"
                    f"YÊU CẦU: {user_request}\n"
                    f"RÀNG BUỘC: {c.to_prompt()}\n"
                    f"CHUYẾN BAY:\n{flights_json}"
                )
            except Exception as e:
                plan_failed = True
                failure_context = f"LLM lỗi khi lên plan: {e}"
                draft = Plan(steps=[], ghi_chu=str(e))

            if not plan_failed:
                if not draft.steps:
                    plan_failed = True
                    failure_context = f"Model nói không có chuyến phù hợp: {draft.ghi_chu}"
                else:
                    # Execute plan
                    self._log(f"⚙️  Thực thi plan ({len(draft.steps)} bước)...")
                    last_pnr = ""

                    for i, step in enumerate(draft.steps, 1):
                        args = {}
                        for k, v in step.args.items():
                            if isinstance(v, str) and v.strip().lower() in ("$pnr", "$booking_code"):
                                args[k] = last_pnr
                            else:
                                args[k] = v

                        result = self._execute_tool_with_harness(step.tool, args)

                        if result.get("status") == "ok" and "pnr" in result:
                            last_pnr = result["pnr"]

                        if result.get("status") not in ("ok",):
                            plan_failed = True
                            failure_context = (
                                f"Step {i}/{len(draft.steps)} thất bại: "
                                f"{step.tool}({args}) → "
                                f"{result.get('reason', result.get('error', result.get('hint', '?')))}"
                            )
                            self._log(f"❌ {failure_context}")
                            break

        # Kiểm tra Phase 1 đã xong chưa
        if self.harness.is_done():
            self._log("✅ Phase 1 thành công!")
            booking = self.harness.get_completed_booking()
            if booking:
                self._final_answer = (
                    f"Đặt vé thành công! PNR: {booking['pnr']}, "
                    f"Chuyến: {booking['ma_chuyen_bay']}, "
                    f"Giá: {booking['gia_moi_ve']:,} VND"
                )
            return self._build_result()

        if not plan_failed:
            # Plan chạy hết nhưng chưa done?
            plan_failed = True
            failure_context = "Plan chạy xong nhưng harness xác nhận chưa hoàn thành."

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: REACT FALLBACK
        # ═══════════════════════════════════════════════════════════
        self._log("━━━ PHASE 2: ReAct Fallback ━━━")
        self._log(f"📌 Lý do chuyển sang ReAct: {failure_context}")

        llm_with_tools = self.llm.bind_tools(self.tools)

        system_msg = REACT_RECOVERY_PROMPT.format(
            failure_context=failure_context,
            constraints=c.to_prompt(),
        )

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=(
                f"Kế hoạch trước đã thất bại: {failure_context}\n\n"
                f"Yêu cầu gốc: {user_request}\n\n"
                f"Hãy thử cách khác để đặt vé cho khách."
            )),
        ]

        for step in range(self.max_react_steps):
            self.total_llm_calls += 1
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                self._log(f"❌ LLM error: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                self._final_answer = response.content
                self._log(f"💬 Model: {response.content[:200]}...")
                break

            for tc in response.tool_calls:
                result = self._execute_tool_with_harness(tc["name"], tc["args"])
                messages.append(ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                ))

                if result.get("error") == "loop_detected":
                    self._log("⛔ Dừng vì lặp trong ReAct phase.")
                    return self._build_result()

            if self.harness.is_done():
                self._log("✅ Phase 2 thành công!")
                self.total_llm_calls += 1
                try:
                    final = llm_with_tools.invoke(messages)
                    self._final_answer = final.content
                except Exception:
                    self._final_answer = "Đã đặt vé thành công (recovery)."
                break

        return self._build_result()
