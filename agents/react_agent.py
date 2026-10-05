"""
agents/react_agent.py — Mẫu thiết kế ReAct (Reasoning + Acting)
═════════════════════════════════════════════════════════════════
ReAct = model quyết định MỘT bước tại một thời điểm.

Vòng lặp:
    ┌────────────────────────────────────────────────────┐
    │  User request                                      │
    │       ↓                                            │
    │  ┌──→ Model suy nghĩ → chọn tool + args ──┐      │
    │  │                                          │      │
    │  │    Harness: kiểm quyền                   │      │
    │  │    ↓ (OK)           ↓ (denied)           │      │
    │  │    Chạy tool        Trả lỗi cho model    │      │
    │  │    ↓                ↓                    │      │
    │  └── observation quay lại model ←───────────┘      │
    │                                                    │
    │  Model nói "xong" → Harness kiểm is_done()        │
    │       ↓ True: THÀNH CÔNG                           │
    │       ↓ False: THẤT BẠI → handoff()                │
    └────────────────────────────────────────────────────┘

ƯU ĐIỂM:
    + Linh hoạt: model thấy kết quả mỗi bước, điều chỉnh được
    + Xử lý được lỗi bất ngờ (tool trả error → model thử cách khác)

NHƯỢC ĐIỂM:
    - Tốn nhiều LLM call (mỗi bước 1 call)
    - Dễ bị lặp nếu không có harness
    - Không preview được kế hoạch trước khi chạy
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)

from .base import BaseFlightAgent


SYSTEM_PROMPT = """Bạn là agent đặt vé máy bay. Nhiệm vụ: giúp khách hàng tìm và đặt vé.

CÁC TOOL CÓ SẴN:
1. search_flight(tu, den, ngay) — Tìm chuyến bay theo chặng và ngày
2. dat_ve(ma_chuyen_bay, so_ve, ten_khach) — Đặt (giữ chỗ) vé, chưa thanh toán
3. thanh_toan(pnr) — Thanh toán cho booking đã giữ chỗ
4. xem_ve(pnr) — Xem lại thông tin booking
5. huy_ve(pnr) — Hủy vé chưa thanh toán
6. hoi_khach_hang(cau_hoi, lua_chon) — Giao tiếp với khách khi có vướng mắc hoặc cần xin ý kiến
7. goi_y_chuyen_bay_thay_the(tu, den, ngay, gia_toi_da) — Tìm chuyến thay thế (ngày khác, giờ khác) khi hết vé
8. xem_chinh_sach_hang(hang) — Tra cứu chính sách hành lý, đổi/hoàn vé

QUY TẮC:
- Luôn search_flight trước để biết có chuyến nào phù hợp
- Chỉ đặt chuyến bay thỏa MỌI ràng buộc của khách (giá, giờ, ngày, chặng)
- Sau khi đặt, phải thanh_toan
- Nếu không có chuyến nào thỏa mãn, có thể gọi goi_y_chuyen_bay_thay_the hoặc hoi_khach_hang để đề xuất phương án nới lỏng

Trả lời bằng tiếng Việt. Khi xong, tóm tắt kết quả cho khách."""


class ReactFlightAgent(BaseFlightAgent):
    """Agent đặt vé máy bay theo mẫu ReAct.

    Model được gọi lặp đi lặp lại, mỗi lần nhận observation từ tool,
    rồi quyết định bước tiếp theo. Harness kiểm soát mỗi tool call.
    """

    pattern_name = "ReAct"

    def run(self, user_request: str) -> dict:
        self.reset()
        self.start_time = time.time()
        self._final_answer = ""

        # Bind tools vào LLM (để LLM biết tool nào có sẵn)
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Khởi tạo messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ]

        self._log(f"📝 User: {user_request}")

        for step in range(self.max_steps):
            # ─── Gọi LLM ───
            self.total_llm_calls += 1
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                self._log(f"❌ LLM error: {e}")
                break

            messages.append(response)

            # ─── Nếu model không gọi tool nào → nó đã "xong" ───
            if not response.tool_calls:
                self._final_answer = response.content
                self._log(f"💬 Model trả lời: {response.content[:200]}...")
                break

            # ─── Xử lý từng tool call ───
            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]
                tc_id = tc["id"]

                # Chạy qua harness
                result = self._execute_tool_with_harness(tool_name, args)

                # Gửi kết quả lại cho model
                tool_msg = ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc_id,
                )
                messages.append(tool_msg)

                # Kiểm tra loop detected → dừng sớm
                if result.get("error") == "loop_detected":
                    self._log("⛔ Dừng vì phát hiện lặp.")
                    return self._build_result()

            # Kiểm tra đã xong chưa (mỗi vòng lặp)
            if self.harness.is_done():
                self._log("✅ Harness xác nhận: ĐÃ HOÀN THÀNH!")
                # Gọi LLM 1 lần nữa để lấy câu trả lời tóm tắt
                self.total_llm_calls += 1
                try:
                    final = llm_with_tools.invoke(messages)
                    self._final_answer = final.content
                except Exception:
                    self._final_answer = "Đã đặt vé thành công."
                break

        return self._build_result()
