# -*- coding: utf-8 -*-
"""
lib/model_gia.py — Model giả lập (Mock LLM) cho Flight Booking Agent
══════════════════════════════════════════════════════════════════
Tương thích hoàn toàn với LangChain BaseChatModel.
Mục đích:
  1. Thay thế API Gemini khi bị Rate Limit (429 ResourceExhausted) hoặc mất mạng.
  2. Kiểm thử độc lập bộ dây đai an toàn (Harness):
     - Ràng buộc là dữ liệu (Constraints as DATA)
     - Kiểm quyền trước khi gọi tool (Permission Check)
     - Kiểm hoàn thành bằng code (Done by Code)
     - Bàn giao có cấu trúc (Handoff to Human)
     - Phát hiện lặp (Loop Detection)
     - Kiểm căn cứ dữ liệu (Grounding Check)
  3. Kiểm thử Tool Calls và vòng lặp cả 3 kiến trúc:
     - ReAct (từng bước qua tool_calls)
     - Plan-then-Execute (qua structured output Plan)
     - Hybrid (Plan trước, ReAct phục hồi khi lỗi)
  4. Chạy tức thì, deterministic, độ tin cậy 100%.

Tham khảo ý tưởng từ SE373 demo2-sv/lib/model_gia.py, mở rộng thêm:
  - Hỗ trợ Flight Booking domain (search, dat_ve, thanh_toan, xem_ve, goi_y...)
  - Hỗ trợ bind_tools() cho ReAct agent
  - Hỗ trợ with_structured_output() cho Planner agent
  - Đa dạng kịch bản kiểm thử: tu_dong, lap, vi_pham_quyen, bip_grounding.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from .tool_call import CHUYEN_BAY, DA_DAT


class ModelGiaFlight(BaseChatModel):
    """Model giả lập tương thích LangChain BaseChatModel cho bài toán đặt vé máy bay.

    Tham số:
        kich_ban: "tu_dong" (mặc định) · "lap" · "vi_pham_quyen" · "bip_grounding"
        so_diem: số lượng vé hoặc bước xử lý tối đa
    """

    kich_ban: str = "tu_dong"
    so_diem: int = 5
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return "se373-flight-model-gia"

    # ─────────────────────────────────────────────────────────────────
    # 1. BIND TOOLS (Hỗ trợ ReAct)
    # ─────────────────────────────────────────────────────────────────
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ModelGiaFlight":
        """LangChain gọi hàm này khi agent khai báo tool. Model giả lập trả về chính nó."""
        return self

    # ─────────────────────────────────────────────────────────────────
    # 2. STRUCTURED OUTPUT (Hỗ trợ Plan-then-Execute và Hybrid)
    # ─────────────────────────────────────────────────────────────────
    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel], Callable, Any],
        **kwargs: Any,
    ) -> Runnable:
        """Hỗ trợ trả về structured output (như Pydantic schema Plan)."""

        def _generate_structured(input_data: Any) -> Any:
            self._luot += 1
            # Chuyển input về dạng text
            if isinstance(input_data, list):
                raw_text = "\n".join(
                    getattr(m, "content", str(m)) for m in input_data
                )
            elif hasattr(input_data, "to_string"):
                raw_text = input_data.to_string()
            else:
                raw_text = str(input_data)

            plan_dict = self._quyet_dinh_plan(raw_text)

            # Nếu schema là Pydantic BaseModel, khởi tạo instance
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema(**plan_dict)
            return plan_dict

        return RunnableLambda(_generate_structured)

    # ─────────────────────────────────────────────────────────────────
    # 3. LLM GENERATE (LangChain invoke)
    # ─────────────────────────────────────────────────────────────────
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._luot += 1
        ai_message = self._quyet_dinh_chat(messages)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ─────────────────────────────────────────────────────────────────
    # 4. TRÍCH XUẤT THÔNG TIN YÊU CẦU TỪ TIN NHẮN / PROMPT
    # ─────────────────────────────────────────────────────────────────
    def _parse_yeu_cau(self, text: str) -> dict:
        """Trích xuất thông tin khách hàng và ràng buộc từ câu lệnh."""
        # 1. Chặng bay: tu -> den
        m_chang = re.search(
            r"chặng\s+([A-Za-z]{3})\s+(?:den|đến|➔|->)\s+([A-Za-z]{3})",
            text,
            re.IGNORECASE,
        )
        tu = m_chang.group(1).upper() if m_chang else "SGN"
        den = m_chang.group(2).upper() if m_chang else "HAN"

        # 2. Ngày bay (YYYY-MM-DD)
        m_ngay = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        ngay = m_ngay.group(1) if m_ngay else "2026-10-01"

        # 3. Giờ muộn nhất (HH:MM)
        m_gio = re.search(r"(?:trước|trước giờ)\s+(\d{1,2}:\d{2})", text, re.IGNORECASE)
        gio = m_gio.group(1) if m_gio else "23:59"
        if len(gio) == 4:  # vd: 7:00 -> 07:00
            gio = "0" + gio

        # 4. Giá tối đa
        m_gia = re.search(
            r"(?:giá tối đa|max|ngân sách)\s+([\d,\.]+)", text, re.IGNORECASE
        )
        if m_gia:
            gia_str = m_gia.group(1).replace(",", "").replace(".", "")
            try:
                gia = int(gia_str)
            except ValueError:
                gia = 2_000_000
        else:
            gia = 2_000_000

        # 5. Số vé
        m_ve = re.search(r"(?:Đặt|so_ve[:=]?)\s+(\d+)\s+vé", text, re.IGNORECASE)
        so_ve = int(m_ve.group(1)) if m_ve else 1

        # 6. Tên khách
        m_khach = re.search(
            r"(?:vé\s+máy\s+bay\s+)?cho\s+([^,\.\n]+?)(?:,|\s+chặng)", text, re.IGNORECASE
        )
        ten_khach = m_khach.group(1).strip() if m_khach else "Nguyen Van A"

        return {
            "tu": tu,
            "den": den,
            "ngay": ngay,
            "gio": gio,
            "gia": gia,
            "so_ve": so_ve,
            "ten_khach": ten_khach,
        }

    # ─────────────────────────────────────────────────────────────────
    # 5. RA QUYẾT ĐỊNH CHO REACT AGENT
    # ─────────────────────────────────────────────────────────────────
    def _quyet_dinh_chat(self, messages: list[BaseMessage]) -> AIMessage:
        """Phân tích chuỗi tin nhắn và quyết định: gọi tool hay kết luận."""
        user_texts = [
            getattr(m, "content", "") for m in messages
            if (isinstance(m, HumanMessage) or getattr(m, "type", "") == "human")
        ]
        target_text = " ".join(user_texts) if user_texts else " ".join(
            getattr(m, "content", "") for m in messages if hasattr(m, "content")
        )
        req = self._parse_yeu_cau(target_text)

        # ── Kịch bản kiểm thử: LẶP (Loop Detector test) ──
        if self.kich_ban == "lap":
            # Liên tục gọi cùng 1 tool với cùng 1 bộ tham số để kích hoạt Harness LoopDetector
            return self._goi(
                "search_flight",
                {"tu": req["tu"], "den": req["den"], "ngay": req["ngay"]},
            )

        # ── Kịch bản kiểm thử: VI PHẠM QUYỀN (Permission Check test) ──
        if self.kich_ban == "vi_pham_quyen":
            # Cố tình gọi dat_ve với chuyến bay có giá 99.000.000 hoặc số vé sai để harness chặn
            return self._goi(
                "dat_ve",
                {"ma_chuyen_bay": "VN204", "so_ve": 99, "ten_khach": "Test Violation"},
            )

        # ── Kịch bản kiểm thử: BỊA ĐẶT DỮ KIỆN (Grounding Check test) ──
        if self.kich_ban == "bip_grounding":
            return AIMessage(
                content="Tôi đã đặt vé thành công: VN9999 PNR9999 giá 99,999,000 VND ngày 2099-01-01 lúc 00:00."
            )

        # ── Kịch bản: TỰ ĐỘNG (chuẩn nghiệp vụ) ──
        # Đọc các quan sát (kết quả tool trả về)
        tool_results = []
        for m in messages:
            if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                tool_results.append(self._doc_json(m.content))

        # Tìm các kết quả cụ thể
        search_res = None
        dat_ve_res = None
        thanh_toan_res = None

        for res in tool_results:
            if not isinstance(res, dict):
                continue
            if "so_ket_qua" in res or "cac_chang_ho_tro" in str(res):
                search_res = res
            elif "pnr" in res and "action" in res and res.get("action") == "thanh_toan":
                dat_ve_res = res
            elif "pnr" in res and res.get("status") == "ok" and res.get("booking", {}).get("da_thanh_toan"):
                thanh_toan_res = res
            elif res.get("status") == "error" and res.get("error") == "het_cho":
                dat_ve_res = res

        # 1. Chưa tìm chuyến bay -> Gọi search_flight
        if not search_res:
            return self._goi(
                "search_flight",
                {"tu": req["tu"], "den": req["den"], "ngay": req["ngay"]},
            )

        # 2. Đã có kết quả search_flight -> Phân tích chuyến bay
        cac_chuyen = search_res.get("ket_qua", [])
        chuyen_thoa_man = [
            cb
            for cb in cac_chuyen
            if cb["tu"] == req["tu"]
            and cb["den"] == req["den"]
            and cb["ngay"] == req["ngay"]
            and cb["gio"] <= req["gio"]
            and cb["gia"] <= req["gia"]
        ]

        # 2.1. Không có chuyến nào thỏa mãn ràng buộc giờ/giá (Testcase NO_FIT)
        if not chuyen_thoa_man:
            return AIMessage(
                content=(
                    f"Rất tiếc, không tìm thấy chuyến bay nào từ {req['tu']} đến {req['den']} "
                    f"ngày {req['ngay']} khởi hành trước {req['gio']} với giá tối đa {req['gia']:,} VND. "
                    f"Các chuyến bay hiện có đều vượt ngân sách hoặc ngoài khung giờ mong muốn. "
                    f"Tôi xin phép bàn giao cho nhân viên tư vấn để hỗ trợ quý khách điều chỉnh kế hoạch."
                )
            )

        # 2.2. Có chuyến thỏa mãn nhưng hết chỗ (Testcase SOLD_OUT)
        chuyen_tot = chuyen_thoa_man[0]
        if chuyen_tot["cho_trong"] < req["so_ve"]:
            # Nếu chưa từng thử đặt chuyến hết chỗ để xác nhận lỗi
            if not dat_ve_res:
                return self._goi(
                    "dat_ve",
                    {
                        "ma_chuyen_bay": chuyen_tot["ma"],
                        "so_ve": req["so_ve"],
                        "ten_khach": req["ten_khach"],
                    },
                )
            # Đã thử đặt và nhận thông báo hết chỗ -> Báo khách và dừng để Handoff
            return AIMessage(
                content=(
                    f"Chuyến bay {chuyen_tot['ma']} ({chuyen_tot['hang']}) từ {req['tu']} đến {req['den']} "
                    f"lúc {chuyen_tot['gio']} thỏa mãn yêu cầu nhưng hiện tại ĐÃ HẾT CHỖ (còn {chuyen_tot['cho_trong']} vé). "
                    f"Tôi xin phép bàn giao cho nhân viên để hỗ trợ quý khách đổi chuyến bay thay thế hoặc chọn ngày khác."
                )
            )

        # 2.3. Chuyến bay thỏa mãn và còn chỗ -> Tiến hành đặt vé
        if not dat_ve_res:
            return self._goi(
                "dat_ve",
                {
                    "ma_chuyen_bay": chuyen_tot["ma"],
                    "so_ve": req["so_ve"],
                    "ten_khach": req["ten_khach"],
                },
            )

        # 3. Đã đặt vé (giữ chỗ thành công) -> Tiến hành thanh toán
        pnr = dat_ve_res.get("pnr")
        if pnr and not thanh_toan_res:
            return self._goi("thanh_toan", {"pnr": pnr})

        # 4. Đã thanh toán xong -> Tóm tắt hoàn tất (chuẩn bị dữ kiện để đạt 100% Grounding)
        if thanh_toan_res:
            booking = thanh_toan_res.get("booking", {})
            return AIMessage(
                content=(
                    f"Tôi đã đặt vé và thanh toán thành công cho hành khách {booking.get('ten_khach', req['ten_khach'])}.\n"
                    f"- Mã chuyến bay: {booking.get('ma_chuyen_bay', chuyen_tot['ma'])}\n"
                    f"- Hãng hàng không: {booking.get('hang', chuyen_tot['hang'])}\n"
                    f"- Hành trình: {booking.get('tu', req['tu'])} đến {booking.get('den', req['den'])}\n"
                    f"- Khởi hành: {booking.get('gio_bay', chuyen_tot['gio'])} ngày {booking.get('ngay_bay', req['ngay'])}\n"
                    f"- Mã đặt chỗ (PNR): {booking.get('pnr', pnr)}\n"
                    f"- Tổng chi phí: {booking.get('tong_tien', chuyen_tot['gia'] * req['so_ve']):,} VND\n"
                    f"Vé đã được thanh toán và xác nhận hoàn tất trên hệ thống!"
                )
            )

        # Fallback an toàn
        return AIMessage(content="Đã xử lý xong các yêu cầu.")

    # ─────────────────────────────────────────────────────────────────
    # 6. RA QUYẾT ĐỊNH CHO PLANNER (Structured Output)
    # ─────────────────────────────────────────────────────────────────
    def _quyet_dinh_plan(self, prompt_text: str) -> dict:
        """Sinh kế hoạch PlanStep cho Plan-then-Execute và Hybrid."""
        req = self._parse_yeu_cau(prompt_text)

        # ── Kịch bản kiểm thử: LẶP ──
        if self.kich_ban == "lap":
            return {
                "steps": [
                    {"tool": "hoi_khach_hang", "args": {"cau_hoi": "Hỏi lần 1", "lua_chon": ["A", "B"]}, "ly_do": "test loop"},
                    {"tool": "hoi_khach_hang", "args": {"cau_hoi": "Hỏi lần 1", "lua_chon": ["A", "B"]}, "ly_do": "test loop"},
                    {"tool": "hoi_khach_hang", "args": {"cau_hoi": "Hỏi lần 1", "lua_chon": ["A", "B"]}, "ly_do": "test loop"},
                    {"tool": "hoi_khach_hang", "args": {"cau_hoi": "Hỏi lần 1", "lua_chon": ["A", "B"]}, "ly_do": "test loop"},
                ],
                "ghi_chu": "Kịch bản cố tình lặp",
            }

        # ── Kịch bản kiểm thử: VI PHẠM QUYỀN ──
        if self.kich_ban == "vi_pham_quyen":
            return {
                "steps": [
                    {"tool": "dat_ve", "args": {"ma_chuyen_bay": "VN204", "so_ve": 99, "ten_khach": "Hacker"}, "ly_do": "vi phạm"},
                ],
                "ghi_chu": "Kịch bản cố tình vi phạm quyền",
            }

        # Đọc các chuyến bay từ prompt nếu có
        cac_chuyen = []
        if "CHUYẾN BAY" in prompt_text:
            try:
                phan_chuyen = prompt_text.split("CHUYẾN BAY")[1]
                # Tìm mảng JSON [...]
                m_json = re.search(r"\[\s*\{.*\}\s*\]", phan_chuyen, re.DOTALL)
                if m_json:
                    cac_chuyen = json.loads(m_json.group(0))
            except Exception:
                pass

        if not cac_chuyen:
            # Lấy từ DB mockup
            cac_chuyen = [
                cb for cb in CHUYEN_BAY
                if cb["tu"] == req["tu"] and cb["den"] == req["den"] and cb["ngay"] == req["ngay"]
            ]

        # Lọc chuyến thỏa mãn
        chuyen_thoa_man = [
            cb for cb in cac_chuyen
            if cb["tu"] == req["tu"]
            and cb["den"] == req["den"]
            and cb["ngay"] == req["ngay"]
            and cb["gio"] <= req["gio"]
            and cb["gia"] <= req["gia"]
        ]

        # 1. Không có chuyến phù hợp (NO_FIT)
        if not chuyen_thoa_man:
            return {
                "steps": [],
                "ghi_chu": f"Không có chuyến bay nào từ {req['tu']} đến {req['den']} thỏa mãn mức giá <= {req['gia']:,} VND và trước {req['gio']}.",
            }

        # 2. Chuyến bay thỏa mãn nhưng hết chỗ (SOLD_OUT)
        chuyen_tot = chuyen_thoa_man[0]
        if chuyen_tot["cho_trong"] < req["so_ve"]:
            return {
                "steps": [],
                "ghi_chu": f"Chuyến bay phù hợp nhất ({chuyen_tot['ma']}) đã hết chỗ (còn {chuyen_tot['cho_trong']} chỗ).",
            }

        # 3. Chuyến bay hợp lệ và còn chỗ -> Lên kế hoạch 3 bước chuẩn
        steps = [
            {
                "tool": "dat_ve",
                "args": {
                    "ma_chuyen_bay": chuyen_tot["ma"],
                    "so_ve": req["so_ve"],
                    "ten_khach": req["ten_khach"],
                },
                "ly_do": f"Giữ chỗ chuyến bay {chuyen_tot['ma']} ({chuyen_tot['hang']}) khởi hành {chuyen_tot['gio']}",
            },
            {
                "tool": "thanh_toan",
                "args": {"pnr": "$pnr"},
                "ly_do": "Thanh toán cho mã đặt chỗ vừa tạo",
            },
            {
                "tool": "xem_ve",
                "args": {"pnr": "$pnr"},
                "ly_do": "Xác nhận lại chi tiết vé máy bay đã thanh toán thành công",
            },
        ]

        return {
            "steps": steps,
            "ghi_chu": f"Kế hoạch đặt vé cho chuyến bay {chuyen_tot['ma']} với giá {chuyen_tot['gia']:,} VND thành công.",
        }

    # ─────────────────────────────────────────────────────────────────
    # 7. TIỆN ÍCH HỖ TRỢ
    # ─────────────────────────────────────────────────────────────────
    def _goi(self, ten: str, args: dict) -> AIMessage:
        cid = f"call_mock_{self._luot}_{ten}"
        tool_call = {"name": ten, "args": args, "id": cid, "type": "tool_call"}
        return AIMessage(content="", tool_calls=[tool_call])

    @staticmethod
    def _doc_json(content: Any) -> Any:
        if isinstance(content, (dict, list)):
            return content
        try:
            return json.loads(content)
        except Exception:
            return content
