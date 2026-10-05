"""
lib/harness.py

Harness kiểm soát quá trình thực thi của Agent:
- Quản lý ràng buộc của người dùng.
- Kiểm tra quyền gọi Tool.
- Xác minh điều kiện hoàn thành.
- Phát hiện vòng lặp và hỗ trợ bàn giao.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage
from .tool_call import CHUYEN_BAY, DA_DAT, xem_ve, TOOL_MAP

# 1. RÀNG BUỘC LÀ DỮ LIỆU
@dataclass
class RangBuoc:
    """Lưu yêu cầu của user thành DATA.

    Model nhận prompt được BUILD TỪ data này.
    """
    tu: str = "SGN"                   # sân bay đi
    den: str = "HAN"                  # sân bay đến
    ngay: str = "2026-10-01"          # ngày bay YYYY-MM-DD
    gio_muon_nhat: str = "23:59"      # bay trước giờ này
    gia_toi_da: int = 2_000_000       # giá tối đa (VND)
    so_ve: int = 1                    # số vé cần đặt
    ten_khach: str = "Nguyen Van A"   # tên hành khách

    def to_prompt(self) -> str:
        """Chuyển ràng buộc thành câu lệnh tự nhiên cho model."""
        return (
            f"Đặt {self.so_ve} vé máy bay cho {self.ten_khach}, "
            f"chặng {self.tu} den {self.den} ngày {self.ngay}, "
            f"khởi hành trước {self.gio_muon_nhat}, "
            f"giá tối đa {self.gia_toi_da:,} VND mỗi vé."
        )

    def chuyen_bay_hop_le(self, cb: dict) -> bool:
        """Kiểm tra 1 chuyến bay có thỏa TẤT CẢ ràng buộc không."""
        return (
            cb.get("tu", cb.get("tu", "")) == self.tu
            and cb.get("den", cb.get("den", "")) == self.den
            and cb.get("ngay", cb.get("ngay_bay", "")) == self.ngay
            and cb.get("gio", cb.get("gio_bay", "")) <= self.gio_muon_nhat
            and cb.get("gia", cb.get("gia_moi_ve", 0)) <= self.gia_toi_da
            and cb.get("cho_trong", 0) >= self.so_ve
        )


# 2. PHÁT HIỆN LẶP (Loop Detector)
class LoopDetector:
    """Phát hiện agent bị lặp: cùng (tool, args) gọi quá N lần.
        Agent đang lặp 1 hành động cùng 1 bộ tham số giống nhau quá N lần → dừng
    """

    def __init__(self, max_repeat: int = 3):
        self.max_repeat = max_repeat
        self.history: list[str] = []      # list of fingerprints

    def _fingerprint(self, tool: str, args: dict) -> str:
        """Tạo dấu vân tay cho 1 tool call."""
        sorted_args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return f"{tool}({sorted_args})"

    def check(self, tool: str, args: dict) -> str | None:
        """Kiểm tra lặp. Trả về cảnh báo nếu phát hiện, None nếu OK."""
        fp = self._fingerprint(tool, args)
        self.history.append(fp)
        count = self.history.count(fp)
        if count >= self.max_repeat:
            return (f"PHÁT HIỆN LẶP: {tool}({args}) đã được gọi {count} lần. "
                    f"Agent đang bị lặp vô hạn.")
        return None

    def reset(self):
        self.history.clear()


# 3. KIỂM CĂN CỨ (Grounding Check)
# Các pattern để trích dữ kiện từ câu trả lời
_MAU_DU_KIEN = [
    re.compile(r"\b[A-Z]{2,3}\d{2,4}\b"),                  # mã chuyến bay: VN204, VJ142
    re.compile(r"\bPNR\d{4}\b"),                             # mã PNR
    re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)"),                    # giá: 1,890,000 hoặc 1.890.000
    re.compile(r"\d{4}-\d{2}-\d{2}"),                        # ngày: 2026-10-01
    re.compile(r"\b\d{2}:\d{2}\b"),                          # giờ: 06:00
]


def kiem_can_cu(cau_tra_loi: str, ket_qua_tool: list[dict]) -> dict:
    """Kiểm tra mọi dữ kiện trong câu trả lời có chính xác với DATA và kết quả tool không.

    Args:
        cau_tra_loi:  Câu trả lời cuối cùng của agent.
        ket_qua_tool: Danh sách kết quả tool đã thu thập.
    Returns:
        {"dat": True/False,
         "chi_tiet": [...],
         "khong_co_nguon": [...]}
    """
    
    if isinstance(cau_tra_loi, list):
        # Trích xuất đoạn text bên trong list các dictionary của Gemini
        cau_tra_loi = " ".join([item.get("text", "") for item in cau_tra_loi if isinstance(item, dict)])
    elif not isinstance(cau_tra_loi, str):
        # Ép kiểu an toàn cho các trường hợp đối tượng lạ khác
        cau_tra_loi = str(cau_tra_loi)
    
    tool_text = json.dumps(ket_qua_tool, ensure_ascii=False)

    du_kien = []
    for mau in _MAU_DU_KIEN:
        du_kien.extend(mau.findall(cau_tra_loi))

    # Chuẩn hóa (bỏ dấu chấm/phẩy phân cách hàng nghìn)
    du_kien_chuan = []
    for dk in du_kien:
        chuan = dk.replace(".", "").replace(",", "")
        du_kien_chuan.append((dk, chuan))

    khong_co_nguon = []
    chi_tiet = []
    for goc, chuan in du_kien_chuan:
        # Tìm cả dạng gốc lẫn chuẩn trong tool_text
        co = (goc in tool_text) or (chuan in tool_text)
        chi_tiet.append({"du_kien": goc, "co_nguon": co})
        if not co:
            khong_co_nguon.append(goc)

    return {
        "dat": len(khong_co_nguon) == 0,
        "chi_tiet": chi_tiet,
        "khong_co_nguon": khong_co_nguon,
    }


# 4-5-6-7. HARNESS CHÍNH
# Chứa các function: kiểm quyền, kiểm hoàn thành, bàn giao, phát hiện lặp, kiểm căn cứ, ... quan trọng
class Harness:
    """Bộ harness chính: kiểm quyền, kiểm hoàn thành, bàn giao.

    Kết hợp tất cả 4 ý tưởng harness:
        1. Ràng buộc là dữ liệu   → self.constraints (RangBuoc)
        2. Kiểm quyền              → self.check_permission()
        3. Kiểm hoàn thành bằng code → self.is_done()
        4. Bàn giao               → self.handoff()
        + Phát hiện lặp           → self.loop_detector
        + Kiểm căn cứ             → self.check_grounding()
    """

    def __init__(self, constraints: RangBuoc, max_repeat: int = 3):
        self.constraints = constraints
        self.loop_detector = LoopDetector(max_repeat=max_repeat)
        self.log: list[tuple[str, dict, dict]] = []   # (tool, args, result)
        self.tool_results: list[dict] = []              # for grounding check

    # ─── 1. RESET — Chuẩn bị cho một phiên chạy hoặc lượt test mới  ───────────────────────────
    def reset(self):
        """Reset harness state (giữ constraints)."""
        self.loop_detector.reset() # Xóa lịch sử "dấu vân tay" để không báo lỗi lặp nhầm từ phiên trước
        self.log.clear() # Dọn sạch nhật ký các lệnh đã gọi
        self.tool_results.clear() # Xóa các kết quả tool cũ để luồng chạy mới bắt đầu từ đầu

    # ─── 2. KIỂM QUYỀN — chạy TRƯỚC tool ───────────────────────────
    def check_permission(self, tool_name: str, args: dict) -> str | None:
        """Trả về None nếu được phép, hoặc lý do nếu bị chặn.

        Logic:
        - dat_ve: chuyến bay phải thỏa constraints
        - thanh_toan: PNR phải tồn tại và chưa thanh toán
        - huy_ve: PNR phải tồn tại
        """
        c = self.constraints

        if tool_name == "dat_ve":
            ma = args.get("ma_chuyen_bay", "").strip().upper()
            chuyen_bay = next((cb for cb in CHUYEN_BAY if cb["ma"] == ma), None)
            if chuyen_bay is None:
                return f"Chuyến bay {ma} không tồn tại."
            if not c.chuyen_bay_hop_le(chuyen_bay):
                vi_pham = []
                if chuyen_bay["tu"] != c.tu or chuyen_bay["den"] != c.den:
                    vi_pham.append(f"sai chặng ({chuyen_bay['tu']}→{chuyen_bay['den']})")
                if chuyen_bay["ngay"] != c.ngay:
                    vi_pham.append(f"sai ngày ({chuyen_bay['ngay']})")
                if chuyen_bay["gio"] > c.gio_muon_nhat:
                    vi_pham.append(f"bay quá muộn ({chuyen_bay['gio']})")
                if chuyen_bay["gia"] > c.gia_toi_da:
                    vi_pham.append(f"quá đắt ({chuyen_bay['gia']:,} VND)")
                if chuyen_bay["cho_trong"] < c.so_ve:
                    vi_pham.append(f"hết chỗ ({chuyen_bay['cho_trong']} còn)")
                return (f"Chuyến {ma} vi phạm ràng buộc: "
                        + "; ".join(vi_pham)
                        + f". Yêu cầu: {c.to_prompt()}")

            # Kiểm tra số vé
            so_ve_yeu_cau = args.get("so_ve", 1)
            if isinstance(so_ve_yeu_cau, str):
                so_ve_yeu_cau = int(so_ve_yeu_cau)
            if so_ve_yeu_cau != c.so_ve:
                return f"Số vé sai: yêu cầu {c.so_ve}, nhưng đặt {so_ve_yeu_cau}."

        if tool_name == "thanh_toan":
            pnr = args.get("pnr", "").strip().upper()
            if pnr not in DA_DAT:
                return f"PNR {pnr} không tồn tại. Đặt vé trước khi thanh toán."
            if DA_DAT[pnr]["da_thanh_toan"]:
                return f"PNR {pnr} đã thanh toán rồi."

        return None  # allowed

    # ─── 3. KIỂM HOÀN THÀNH — đọc lại từ hệ thống ─────────────────
    def is_done(self) -> bool:
        """Done = có ít nhất 1 booking đã thanh toán thỏa constraints.

        QUAN TRỌNG: Đọc LẠI từ hệ thống, không tin model.
        """
        c = self.constraints
        for pnr in DA_DAT:
            result = xem_ve(pnr)
            if result["status"] != "ok":
                continue
            b = result["booking"]
            if (b["da_thanh_toan"]
                and b["tu"] == c.tu
                and b["den"] == c.den
                and b["ngay_bay"] == c.ngay
                and b["gio_bay"] <= c.gio_muon_nhat
                and b["gia_moi_ve"] <= c.gia_toi_da
                and b["so_ve"] >= c.so_ve):
                return True
        return False

    def get_completed_booking(self) -> dict | None:
        """Trả về booking đã hoàn thành (nếu có)."""
        c = self.constraints
        for pnr in DA_DAT:
            result = xem_ve(pnr)
            if result["status"] != "ok":
                continue
            b = result["booking"]
            if (b["da_thanh_toan"]
                and b["tu"] == c.tu and b["den"] == c.den
                and b["ngay_bay"] == c.ngay and b["gio_bay"] <= c.gio_muon_nhat
                and b["gia_moi_ve"] <= c.gia_toi_da and b["so_ve"] >= c.so_ve):
                return b
        return None

    # ─── 4. BÀN GIAO — 3 thứ cho con người ─────────────────────────
    def handoff(self, cau_hoi: str = "") -> dict:
        """Tạo báo cáo bàn giao khi agent thất bại.

        Returns:
            {
                "da_lam":   ["PNR0001: chưa thanh toán", ...],
                "da_thu":   ["search_flight({...}) → ok", ...],
                "cau_hoi":  "Không tìm thấy chuyến bay... nới lỏng ràng buộc?"
            }
        """
        da_lam = []
        for pnr, b in DA_DAT.items():
            tt = "đã thanh toán" if b["da_thanh_toan"] else "chưa thanh toán"
            da_lam.append(f"{pnr} ({b['ma_chuyen_bay']}): {tt}")

        da_thu = []
        for tool, args, result in self.log:
            status = result.get("status", "?")
            da_thu.append(f"{tool}({json.dumps(args, ensure_ascii=False)}) → {status}")

        if not cau_hoi:
            cau_hoi = ("Không có chuyến bay nào thỏa tất cả ràng buộc. "
                       "Bạn muốn nới lỏng điều kiện nào: "
                       "giờ khởi hành, giá tối đa, hay ngày bay?")

        return {
            "da_lam": da_lam or ["Chưa đặt vé nào."],
            "da_thu": da_thu or ["Chưa thử gì."],
            "cau_hoi": cau_hoi,
        }

    # ─── LOG & GROUNDING ────────────────────────────────────────────
    def log_call(self, tool: str, args: dict, result: dict):
        """Ghi nhận 1 tool call vào log."""
        self.log.append((tool, args, result))
        self.tool_results.append(result)

    def check_grounding(self, answer: str) -> dict:
        """Kiểm tra căn cứ cho câu trả lời cuối cùng."""
        return kiem_can_cu(answer, self.tool_results)

    # ─── TIỆN ÍCH ───────────────────────────────────────────────────
    def get_trace(self) -> str:
        """Trả về trace dạng text, dễ đọc."""
        lines = []
        for i, (tool, args, result) in enumerate(self.log, 1):
            status = result.get("status", "?")
            reason = result.get("reason", result.get("message", ""))
            if not reason and "hint" in result:
                hint = result["hint"]
                reason = hint.get("message", "") if isinstance(hint, dict) else str(hint)
            lines.append(f"[{i}] {tool}({json.dumps(args, ensure_ascii=False)}) "
                         f"→ {status}" + (f" | {reason}" if reason else ""))
        return "\n".join(lines)

    def get_result(self) -> dict:
        """Trả về kết quả cuối cùng: DONE hoặc FAILED + handoff."""
        if self.is_done():
            booking = self.get_completed_booking()
            return {"ket_qua": "THANH_CONG", "booking": booking}
        else:
            return {"ket_qua": "THAT_BAI", "ban_giao": self.handoff()}

    # ─── 5. MIDDLEWARE TÍCH HỢP LANGCHAIN (ĐIỀU KIỆN DỪNG & KIỂM QUYỀN) ───
    def create_tool_middleware(self):
        """Tạo middleware chặn tool call theo chuẩn của LangChain (@wrap_tool_call).
        
        Kiểm quyền và phát hiện lặp trước khi cho phép tool thực thi.
        """
        @wrap_tool_call
        def _tool_middleware(request, handler):
            call = request.tool_call
            name = call["name"]
            args = call["args"]

            # 1. Kiểm tra lặp (Loop Detection)
            loop_warning = self.loop_detector.check(name, args)
            if loop_warning:
                result = {"status": "denied", "reason": loop_warning, "error": "loop_detected"}
                self.log_call(name, args, result)
                return ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"])

            # 2. Kiểm quyền (Permission Check)
            reason = self.check_permission(name, args)
            if reason:
                result = {"status": "denied", "reason": reason}
            else:
                try:
                    fn = TOOL_MAP.get(name)
                    result = fn(**args) if fn else {"status": "error", "error": f"Tool '{name}' không tồn tại"}
                except Exception as e:
                    result = {"status": "error", "error": str(e)}

            self.log_call(name, args, result)
            return ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"])

        return _tool_middleware

    def get_middlewares(self, run_limit: int = 10) -> list:
        """Trả về danh sách middleware chuẩn của LangChain:
        1. Tool call wrapper (@wrap_tool_call: kiểm quyền + phát hiện lặp)
        2. ModelCallLimitMiddleware (kiểm soát trần số lần gọi model / dừng xác định theo ngân sách)
        """
        return [
            self.create_tool_middleware(),
            ModelCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"),
        ]
