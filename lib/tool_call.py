"""
lib/tool_call.py — Dữ liệu mockup & các công cụ (Tools) chuyến bay
═════════════════════════════════════════════════════════════════════
Bao gồm:
  1. Cơ sở dữ liệu giả lập (CHUYEN_BAY, DA_DAT).
  2. 5 Tool cốt lõi: search_flight, dat_ve, thanh_toan, xem_ve, huy_ve.
  3. 3 Tool hỗ trợ khi gặp lỗi/ngoại lệ:
     - hoi_khach_hang: Giao tiếp trực tiếp với khách (Human-in-the-loop)
     - goi_y_chuyen_bay_thay_the: Tìm chuyến thay thế ngày lân cận/giờ khác
     - xem_chinh_sach_hang: Tra cứu hành lý, đổi/hoàn vé
"""

from datetime import datetime

# 1. BẢNG DỮ LIỆU CHUYẾN BAY MẪU
CHUYEN_BAY = [
    # --- SGN → HAN ---
    dict(ma="VN204",  tu="SGN", den="HAN", ngay="2026-10-01", gio="06:00",
         gia=1_890_000, hang="Vietnam Airlines",  cho_trong=4),
    dict(ma="VJ142",  tu="SGN", den="HAN", ngay="2026-10-01", gio="09:30",
         gia=1_250_000, hang="Vietjet Air",       cho_trong=0),   # hết chỗ
    dict(ma="VJ148",  tu="SGN", den="HAN", ngay="2026-10-01", gio="23:50",
         gia=  990_000, hang="Vietjet Air",       cho_trong=8),
    dict(ma="QH202",  tu="SGN", den="HAN", ngay="2026-10-02", gio="14:15",
         gia=1_690_000, hang="Bamboo Airways",    cho_trong=7),
    dict(ma="VN206",  tu="SGN", den="HAN", ngay="2026-10-03", gio="06:00",
         gia=1_950_000, hang="Vietnam Airlines",  cho_trong=5),
    # --- SGN → DAD ---
    dict(ma="VN109",  tu="SGN", den="DAD", ngay="2026-10-01", gio="07:20",
         gia=1_120_000, hang="Vietnam Airlines",  cho_trong=3),
    dict(ma="VJ501",  tu="SGN", den="DAD", ngay="2026-10-02", gio="12:00",
         gia=  890_000, hang="Vietjet Air",       cho_trong=9),
    # --- HAN → DAD ---
    dict(ma="VN266",  tu="HAN", den="DAD", ngay="2026-10-01", gio="08:00",
         gia=1_050_000, hang="Vietnam Airlines",  cho_trong=2),
    # --- SGN → PQC ---
    dict(ma="BL801",  tu="SGN", den="PQC", ngay="2026-10-02", gio="10:10",
         gia=  980_000, hang="Pacific Airlines",  cho_trong=5),
    # --- SGN → CXR ---
    dict(ma="VN1360", tu="SGN", den="CXR", ngay="2026-10-01", gio="15:40",
         gia=1_150_000, hang="Vietnam Airlines",  cho_trong=0),   # hết chỗ
    dict(ma="VN1362", tu="SGN", den="CXR", ngay="2026-10-03", gio="15:40",
         gia=1_190_000, hang="Vietnam Airlines",  cho_trong=6),
]

# "DATABASE" ĐẶT VÉ — dict[pnr_code → booking_info]
DA_DAT: dict[str, dict] = {}
_count_pnr = 0


def reset_db():
    """Khôi phục lại toàn bộ dữ liệu ban đầu (dùng khi chạy test nhiều lần)."""
    global DA_DAT, _count_pnr
    DA_DAT.clear()
    _count_pnr = 0
    # Khôi phục số ghế trống gốc
    cho_goc = {
        "VN204": 4, "VJ142": 0, "VJ148": 8, "QH202": 7, "VN206": 5,
        "VN109": 3, "VJ501": 9, "VN266": 2, "BL801": 5,
        "VN1360": 0, "VN1362": 6,
    }
    for cb in CHUYEN_BAY:
        cb["cho_trong"] = cho_goc.get(cb["ma"], cb["cho_trong"])


# 2. CÁC TOOL CỐT LÕI (CORE TOOLS)

def search_flight(tu: str, den: str, ngay: str = "") -> dict:
    """Tìm kiếm chuyến bay theo điểm đi (tu), điểm đến (den) và ngày bay (YYYY-MM-DD, tùy chọn).
    
    Args:
        tu: Mã sân bay đi (VD: SGN, HAN, DAD)
        den: Mã sân bay đến (VD: HAN, DAD, CXR)
        ngay: Ngày khởi hành dạng YYYY-MM-DD (bỏ trống nếu muốn tìm tất cả ngày)
    """
    tu, den = tu.strip().upper(), den.strip().upper()
    ngay = ngay.strip()

    ket_qua = [
        cb for cb in CHUYEN_BAY
        if cb["tu"] == tu and cb["den"] == den
        and (not ngay or cb["ngay"] == ngay)
    ]

    if not ket_qua:
        cac_chang = sorted({f"{cb['tu']}-{cb['den']}" for cb in CHUYEN_BAY})
        return {
            "status": "error",
            "error": "khong_co_chuyen_bay",
            "message": f"Không tìm thấy chuyến bay {tu} → {den}" + (f" ngày {ngay}" if ngay else "") + ".",
            "hint": {
                "action": "goi_y_chuyen_bay_thay_the",
                "message": "Không có chuyến chính xác ngày này. Gọi goi_y_chuyen_bay_thay_the hoặc thử chặng khác.",
                "data": {"cac_chang_ho_tro": cac_chang}
            }
        }

    return {"status": "ok", "so_ket_qua": len(ket_qua), "ket_qua": ket_qua}


def dat_ve(ma_chuyen_bay: str, so_ve: int, ten_khach: str) -> dict:
    """Đặt vé (giữ chỗ) cho chuyến bay. Thao tác này chưa thanh toán tiền.
    
    Args:
        ma_chuyen_bay: Mã chuyến bay (VD: VN204, VJ148)
        so_ve: Số lượng vé cần đặt (phải >= 1)
        ten_khach: Họ và tên hành khách
    """
    global _count_pnr
    ma_chuyen_bay = ma_chuyen_bay.strip().upper()
    ten_khach = ten_khach.strip()

    chuyen_bay = next((cb for cb in CHUYEN_BAY if cb["ma"] == ma_chuyen_bay), None)
    if not chuyen_bay:
        cac_ma = [cb["ma"] for cb in CHUYEN_BAY]
        return {
            "status": "error",
            "error": "khong_tim_thay",
            "message": f"Mã chuyến {ma_chuyen_bay} không tồn tại.",
            "hint": {
                "action": "search_flight",
                "message": "Xác định lại mã chuyến bay qua search_flight.",
                "data": {"ma_chuyen_bay_nhap": ma_chuyen_bay, "danh_sach_ma_hop_le": cac_ma}
            }
        }

    if so_ve <= 0:
        return {
            "status": "error",
            "error": "so_ve_khong_hop_le",
            "message": "Số vé phải >= 1.",
            "hint": {
                "action": "dat_ve",
                "message": "Vui lòng nhập số vé là số nguyên dương.",
                "data": {"ma_chuyen_bay": ma_chuyen_bay, "ten_khach": ten_khach}
            }
        }

    if chuyen_bay["cho_trong"] < so_ve:
        # Tìm chuyến cùng chặng còn chỗ để gợi ý
        con_cho = [cb["ma"] for cb in CHUYEN_BAY 
                   if cb["tu"] == chuyen_bay["tu"] and cb["den"] == chuyen_bay["den"] 
                   and cb["cho_trong"] >= so_ve and cb["ma"] != ma_chuyen_bay]
        return {
            "status": "error",
            "error": "het_cho",
            "message": f"Chuyến {ma_chuyen_bay} chỉ còn {chuyen_bay['cho_trong']} chỗ (cần {so_ve} chỗ).",
            "hint": {
                "action": "goi_y_chuyen_bay_thay_the",
                "message": "Chuyến này hết chỗ. Gọi goi_y_chuyen_bay_thay_the hoặc hoi_khach_hang.",
                "data": {"cac_chuyen_cung_chang_con_cho": con_cho}
            },
            "ma": ma_chuyen_bay,
            "cho_trong": chuyen_bay["cho_trong"]
        }

    # Giữ chỗ thành công
    chuyen_bay["cho_trong"] -= so_ve
    _count_pnr += 1
    pnr = f"PNR{_count_pnr:04d}"

    DA_DAT[pnr] = {
        "pnr": pnr,
        "ma_chuyen_bay": ma_chuyen_bay,
        "ten_khach": ten_khach,
        "so_ve": so_ve,
        "gia_moi_ve": chuyen_bay["gia"],
        "tong_tien": chuyen_bay["gia"] * so_ve,
        "da_thanh_toan": False,
        "ngay_dat": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tu": chuyen_bay["tu"],
        "den": chuyen_bay["den"],
        "ngay_bay": chuyen_bay["ngay"],
        "gio_bay": chuyen_bay["gio"],
        "hang": chuyen_bay["hang"],
    }

    return {"status": "ok", 
            "message": f"Đặt vé thành công. Mã PNR: {pnr}. Vui lòng thanh toán để xác nhận.",
            "action": "thanh_toan",
            "pnr": pnr, "booking": DA_DAT[pnr]}


def thanh_toan(pnr: str) -> dict:
    """Thanh toán tiền cho một mã đặt chỗ (PNR) đã giữ chỗ. Thao tác này không thể hoàn tác.
    
    Args:
        pnr: Mã đặt chỗ PNR (VD: PNR0001)
    """
    pnr = pnr.strip().upper()
    if pnr not in DA_DAT:
        return {
            "status": "error",
            "error": "khong_tim_thay_pnr",
            "message": f"Mã PNR '{pnr}' không tồn tại trong hệ thống.",
            "hint": {
                "action": "dat_ve",
                "message": "Chưa có booking nào được tạo. Cần gọi dat_ve trước khi thanh toán.",
                "data": {"pnr_nhap": pnr}
            }
        }

    if DA_DAT[pnr]["da_thanh_toan"]:
        return {
            "status": "error",
            "error": "da_thanh_toan_roi",
            "message": f"PNR '{pnr}' đã được thanh toán trước đó rồi.",
            "hint": {
                "action": "xem_ve",
                "message": "Không cần thanh toán lại. Gọi xem_ve để kiểm tra chi tiết vé.",
                "data": {"pnr": pnr}
            }
        }

    DA_DAT[pnr]["da_thanh_toan"] = True
    return {"status": "ok", "pnr": pnr, "booking": DA_DAT[pnr]}


def xem_ve(pnr: str) -> dict:
    """Đọc lại toàn bộ thông tin chi tiết của booking từ hệ thống bằng mã PNR.
    
    Args:
        pnr: Mã đặt chỗ PNR (VD: PNR0001)
    """
    pnr = pnr.strip().upper()
    if pnr not in DA_DAT:
        return {
            "status": "error",
            "error": "khong_tim_thay_pnr",
            "message": f"Mã PNR '{pnr}' không tồn tại.",
            "hint": {
                "action": "dat_ve",
                "message": "PNR không tồn tại. Kiểm tra lại mã hoặc tiến hành dat_ve mới.",
                "data": {"pnr": pnr}
            }
        }
    return {"status": "ok", "booking": DA_DAT[pnr]}


def huy_ve(pnr: str) -> dict:
    """Hủy một booking đang giữ chỗ chưa thanh toán. Vé đã thanh toán không thể hủy bằng hàm này.
    
    Args:
        pnr: Mã đặt chỗ PNR (VD: PNR0001)
    """
    pnr = pnr.strip().upper()
    if pnr not in DA_DAT:
        return {
            "status": "error",
            "error": "khong_tim_thay_pnr",
            "message": f"Mã PNR '{pnr}' không tồn tại.",
            "hint": {
                "action": "xem_ve",
                "message": "Không tìm thấy vé để hủy. Kiểm tra lại mã PNR.",
                "data": {"pnr": pnr}
            }
        }

    booking = DA_DAT[pnr]
    if booking["da_thanh_toan"]:
        return {
            "status": "error",
            "error": "da_thanh_toan",
            "message": f"PNR '{pnr}' đã thanh toán tiền. Không thể hủy vé đã thanh toán qua tool này.",
            "hint": {
                "action": "hoi_khach_hang",
                "message": "Vé đã thanh toán. Hỏi khách hàng xem có muốn gửi yêu cầu hoàn vé đặc biệt không.",
                "data": {"pnr": pnr}
            }
        }

    # Hoàn trả chỗ trống
    chuyen_bay = next((cb for cb in CHUYEN_BAY if cb["ma"] == booking["ma_chuyen_bay"]), None)
    if chuyen_bay:
        chuyen_bay["cho_trong"] += booking["so_ve"]

    del DA_DAT[pnr]
    return {"status": "ok", "pnr": pnr, "message": f"Đã hủy vé giữ chỗ {pnr} thành công."}


# 3. CÁC TOOL HỖ TRỢ KHI GẶP LỖI / NGOẠI LỆ (HELPER & RECOVERY TOOLS)

def hoi_khach_hang(cau_hoi: str, lua_chon: list[str] = None) -> dict:
    """Giao tiếp trực tiếp với khách hàng khi gặp bế tắc, ràng buộc mâu thuẫn hoặc cần người dùng xác nhận quyết định (Human-in-the-loop).
    
    Args:
        cau_hoi: Câu hỏi rõ ràng, ngắn gọn muốn hỏi khách hàng
        lua_chon: Danh sách các phương án gợi ý cho khách chọn nhanh (tùy chọn)
    """
    if lua_chon is None:
        lua_chon = ["Đồng ý", "Từ chối", "Chọn phương án khác"]

    return {
        "status": "ok",
        "action_required": "cho_khach_tra_loi",
        "cau_hoi": cau_hoi,
        "lua_chon": lua_chon,
        "message": f"Đã gửi câu hỏi tới khách hàng: '{cau_hoi}'",
        # Mô phỏng phản hồi từ khách hàng cho hệ thống tự động:
        "phan_hoi_mac_dinh": "Khách hàng đồng ý nới lỏng điều kiện giờ bay hoặc xem xét chuyến thay thế tốt nhất."
    }


def goi_y_chuyen_bay_thay_the(tu: str, den: str, ngay: str, gia_toi_da: int = 0) -> dict:
    """Tìm kiếm các chuyến bay thay thế trên cùng chặng khi ngày chỉ định không có vé hoặc bị hết chỗ. Quét các ngày lân cận và các chuyến còn chỗ.
    
    Args:
        tu: Mã sân bay đi
        den: Mã sân bay đến
        ngay: Ngày gốc khách muốn đi (YYYY-MM-DD)
        gia_toi_da: Mức giá tối đa mong muốn (0 = không giới hạn giá)
    """
    tu, den = tu.strip().upper(), den.strip().upper()
    
    # Tìm tất cả chuyến cùng chặng còn chỗ
    cung_chang = [
        cb for cb in CHUYEN_BAY
        if cb["tu"] == tu and cb["den"] == den and cb["cho_trong"] > 0
    ]

    if not cung_chang:
        return {
            "status": "error",
            "message": f"Chặng {tu} → {den} hiện tại không còn chuyến nào trống chỗ trên toàn hệ thống.",
            "hint": {
                "action": "hoi_khach_hang", 
                "message": "Thông báo cho khách và hỏi đổi chặng bay."}
        }

    # Phân loại: cùng ngày khác giờ, hoặc ngày khác
    cung_ngay = [cb for cb in cung_chang if cb["ngay"] == ngay]
    khac_ngay = [cb for cb in cung_chang if cb["ngay"] != ngay]

    goi_y = []
    if cung_ngay:
        goi_y.append({"loai": "cung_ngay_gio_khac", "danh_sach": cung_ngay})
    if khac_ngay:
        goi_y.append({"loai": "ngay_lan_can", "danh_sach": khac_ngay})

    return {
        "status": "ok",
        "tu": tu,
        "den": den,
        "ngay_goc": ngay,
        "so_phuong_an": len(cung_chang),
        "phuong_an_thay_the": goi_y,
        "message": f"Tìm thấy {len(cung_chang)} chuyến bay thay thế còn chỗ trên chặng {tu} → {den}."
    }


def xem_chinh_sach_hang(hang: str) -> dict:
    """Tra cứu chính sách hành lý xách tay, hành lý ký gửi và điều kiện đổi/hoàn vé của hãng hàng không.
    
    Args:
        hang: Tên hãng hàng không (VD: Vietnam Airlines, Vietjet Air, Bamboo Airways, Pacific Airlines)
    """
    hang_lower = hang.strip().lower()

    CHINH_SACH = {
        "vietnam airlines": {
            "hang": "Vietnam Airlines",
            "hanh_ly_xach_tay": "12kg (1 kiện + 1 phụ kiện)",
            "hanh_ly_ky_gui": "23kg miễn phí (hạng phổ thông tiêu chuẩn)",
            "doi_ngay_bay": "Miễn phí trước 24h (chỉ thu chênh lệch giá nếu có)",
            "hoan_ve": "Có hỗ trợ (phí từ 350.000 - 500.000 VND tùy hạng vé)"
        },
        "vietjet air": {
            "hang": "Vietjet Air",
            "hanh_ly_xach_tay": "7kg tiêu chuẩn",
            "hanh_ly_ky_gui": "Chưa bao gồm (cần mua thêm gói 15kg-40kg)",
            "doi_ngay_bay": "Được phép đổi trước 3h khởi hành, phí đổi 370.000 VND + chênh lệch",
            "hoan_ve": "Không hoàn tiền mặt, bảo lưu định danh 180 ngày"
        },
        "bamboo airways": {
            "hang": "Bamboo Airways",
            "hanh_ly_xach_tay": "7kg",
            "hanh_ly_ky_gui": "20kg miễn phí cho hạng vé Economy",
            "doi_ngay_bay": "Phí đổi 360.000 VND + chênh lệch giá",
            "hoan_ve": "Tùy hạng vé (hạng linh hoạt cho hoàn phí 450.000 VND)"
        },
        "pacific airlines": {
            "hang": "Pacific Airlines",
            "hanh_ly_xach_tay": "7kg",
            "hanh_ly_ky_gui": "Mua thêm theo gói",
            "doi_ngay_bay": "Được phép đổi có thu phí",
            "hoan_ve": "Không hỗ trợ hoàn vé tiết kiệm"
        }
    }

    for key, info in CHINH_SACH.items():
        if key in hang_lower or hang_lower in key:
            return {"status": "ok", "chinh_sach": info}

    return {
        "status": "error",
        "error": "khong_tim_thay_hang",
        "message": f"Chưa có dữ liệu chính sách cho hãng '{hang}'.",
        "cac_hang_co_san": list(CHINH_SACH.keys())
    }


# 4. DANH SÁCH TẤT CẢ CÁC TOOL CUNG CẤP CHO AGENT
ALL_TOOLS = [
    # 5 tool nghiệp vụ chính
    search_flight, dat_ve, thanh_toan, xem_ve, huy_ve,
    # 3 tool hỗ trợ xử lý lỗi & giao tiếp
    hoi_khach_hang, goi_y_chuyen_bay_thay_the, xem_chinh_sach_hang
]

TOOL_MAP = {fn.__name__: fn for fn in ALL_TOOLS}
"""Ánh xạ tên chuỗi -> hàm thực thi để Harness và Agent gọi."""