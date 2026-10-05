"""
evaluate.py — Đánh giá hiệu quả 3 mẫu thiết kế Agent
═══════════════════════════════════════════════════════════
Chạy mỗi agent trên nhiều kịch bản test, so sánh:
    - Tỷ lệ thành công
    - Số lượng LLM calls
    - Số lượng tool calls
    - Thời gian chạy
    - Chất lượng bàn giao (khi thất bại)

Kịch bản test:
    1. EASY:    có chuyến bay thỏa mọi ràng buộc → phải thành công
    2. NO_FIT:  không có chuyến nào thỏa mọi ràng buộc → phải thất bại + handoff
    3. SOLD_OUT: chuyến duy nhất thỏa mọi ràng buộc nhưng hết chỗ → phải thất bại

Chạy:
    python evaluate.py
"""

import json
import time
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tabulate import tabulate

from lib.harness import RangBuoc
from agents.react_agent import ReactFlightAgent
from agents.plan_execute_agent import PlanExecuteFlightAgent
from agents.hybrid_agent import HybridFlightAgent


# ═════════════════════════════════════════════════════════════════════
# CÁC KỊCH BẢN TEST
# ═════════════════════════════════════════════════════════════════════
TEST_CASES = [
    {
        "ten": "EASY — Có chuyến phù hợp",
        "mo_ta": "SGN→HAN ngày 01/10, giá max 2M, trước 07:00. VN204 thỏa mãn.",
        "constraints": RangBuoc(
            tu="SGN", den="HAN", ngay="2026-10-01",
            gio_muon_nhat="07:00", gia_toi_da=2_000_000,
            so_ve=1, ten_khach="Nguyen Van A",
        ),
        "ket_qua_mong_doi": "THANH_CONG",
    },
    {
        "ten": "NO_FIT — Không có chuyến phù hợp",
        "mo_ta": "SGN→HAN ngày 01/10, giá max 1M, trước 10:00. "
                 "VN204=1.89M (quá đắt), VJ142=1.25M (quá đắt). Không chuyến nào thỏa.",
        "constraints": RangBuoc(
            tu="SGN", den="HAN", ngay="2026-10-01",
            gio_muon_nhat="10:00", gia_toi_da=1_000_000,
            so_ve=1, ten_khach="Tran Thi B",
        ),
        "ket_qua_mong_doi": "THAT_BAI",
    },
    {
        "ten": "SOLD_OUT — Hết chỗ",
        "mo_ta": "SGN→CXR ngày 01/10. VN1360 thỏa nhưng hết chỗ.",
        "constraints": RangBuoc(
            tu="SGN", den="CXR", ngay="2026-10-01",
            gio_muon_nhat="23:59", gia_toi_da=2_000_000,
            so_ve=1, ten_khach="Le Van C",
        ),
        "ket_qua_mong_doi": "THAT_BAI",
    },
]


def run_evaluation(model_name: str = "gemini-3.8-flash", verbose: bool = True, no_api: bool = False):
    """Chạy đánh giá tất cả 3 agent patterns trên tất cả test cases."""

    if no_api or model_name.lower() in ("mock", "model_gia", "fake"):
        model_name = "mock"
        print("\n⚡ [CHẾ ĐỘ NO-API] Sử dụng Model Giả Lập — Không tốn API Key & Tránh Rate Limit!")

    agent_classes = [
        ("ReAct",            ReactFlightAgent),
        ("Plan-then-Execute", PlanExecuteFlightAgent),
        ("Hybrid",           HybridFlightAgent),
    ]

    # Bảng kết quả
    all_results = []

    for tc in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"📋 Kịch bản: {tc['ten']}")
        print(f"   {tc['mo_ta']}")
        print(f"   Mong đợi: {tc['ket_qua_mong_doi']}")
        print(f"{'='*70}")

        for pattern_name, AgentClass in agent_classes:
            print(f"\n  ── {pattern_name} ──")
            constraints = tc["constraints"]
            user_request = constraints.to_prompt()

            agent = AgentClass(
                constraints=constraints,
                model_name=model_name,
                verbose=verbose,
            )

            try:
                result = agent.run(user_request)
            except Exception as e:
                result = {
                    "ket_qua": "LOI",
                    "metrics": {
                        "pattern": pattern_name,
                        "llm_calls": agent.total_llm_calls,
                        "tool_calls": agent.total_tool_calls,
                        "time_seconds": 0,
                    },
                    "trace": f"ERROR: {e}",
                }

            # Đánh giá
            ket_qua_thuc = result.get("ket_qua", "LOI")
            dung = ket_qua_thuc == tc["ket_qua_mong_doi"]
            metrics = result.get("metrics", {})

            # Kiểm tra bàn giao (nếu thất bại, phải có handoff đầy đủ)
            handoff_ok = True
            if ket_qua_thuc == "THAT_BAI":
                ban_giao = result.get("ban_giao", {})
                handoff_ok = bool(ban_giao and ban_giao.get("da_thu") and ban_giao.get("cau_hoi"))

            row = {
                "Kịch bản": tc["ten"].split(" — ")[0],
                "Pattern": pattern_name,
                "Kết quả": ket_qua_thuc,
                "Đúng?": "✅" if dung else "❌",
                "LLM calls": metrics.get("llm_calls", "?"),
                "Tool calls": metrics.get("tool_calls", "?"),
                "Thời gian (s)": metrics.get("time_seconds", "?"),
                "Handoff OK?": "✅" if handoff_ok else "❌" if ket_qua_thuc == "THAT_BAI" else "—",
            }
            all_results.append(row)

            # In trace
            if verbose:
                print(f"    Kết quả: {ket_qua_thuc} {'✅ ĐÚNG' if dung else '❌ SAI'}")
                print(f"    LLM calls: {metrics.get('llm_calls', '?')}, "
                      f"Tool calls: {metrics.get('tool_calls', '?')}, "
                      f"Thời gian: {metrics.get('time_seconds', '?')}s")
                if result.get("trace"):
                    print(f"    Trace:\n{result['trace']}")

    # ═════════════════════════════════════════════════════════════
    # BẢNG TỔNG KẾT
    # ═════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("📊 BẢNG TỔNG KẾT ĐÁNH GIÁ")
    print(f"{'='*70}\n")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))

    # Tổng hợp theo pattern
    print(f"\n{'─'*50}")
    print("📈 TỔNG HỢP THEO PATTERN:")
    print(f"{'─'*50}")

    for pattern_name, _ in agent_classes:
        rows = [r for r in all_results if r["Pattern"] == pattern_name]
        dung_count = sum(1 for r in rows if r["Đúng?"] == "✅")
        total = len(rows)
        avg_llm = sum(r["LLM calls"] for r in rows if isinstance(r["LLM calls"], (int, float))) / max(total, 1)
        avg_tool = sum(r["Tool calls"] for r in rows if isinstance(r["Tool calls"], (int, float))) / max(total, 1)
        avg_time = sum(r["Thời gian (s)"] for r in rows if isinstance(r["Thời gian (s)"], (int, float))) / max(total, 1)

        print(f"\n  {pattern_name}:")
        print(f"    Tỷ lệ đúng: {dung_count}/{total} ({dung_count/total*100:.0f}%)")
        print(f"    Trung bình LLM calls: {avg_llm:.1f}")
        print(f"    Trung bình Tool calls: {avg_tool:.1f}")
        print(f"    Trung bình thời gian: {avg_time:.1f}s")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá 3 mẫu thiết kế agent")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Tên model Gemini")
    parser.add_argument("--no-api", action="store_true", help="Sử dụng model giả lập (không cần API key, tránh rate limit)")
    parser.add_argument("--quiet", action="store_true", help="Không in chi tiết")
    args = parser.parse_args()

    run_evaluation(model_name=args.model, verbose=not args.quiet, no_api=args.no_api)
