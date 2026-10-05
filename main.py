"""
main.py — Entry point: Chạy thử agent đặt vé máy bay
═════════════════════════════════════════════════════
Chạy:
    python main.py                    # Chạy tương tác (chọn pattern, nhập yêu cầu)
    python main.py --pattern react    # Chỉ chạy ReAct
    python main.py --pattern plan     # Chỉ chạy Plan-then-Execute
    python main.py --pattern hybrid   # Chỉ chạy Hybrid
    python main.py --evaluate         # Chạy đánh giá 3 pattern
"""

import json
import argparse

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from lib.harness import RangBuoc
from agents.react_agent import ReactFlightAgent
from agents.plan_execute_agent import PlanExecuteFlightAgent
from agents.hybrid_agent import HybridFlightAgent
from agents.base import MockFlightAgent


PATTERNS = {
    "react": ("ReAct", ReactFlightAgent),
    "plan": ("Plan-then-Execute", PlanExecuteFlightAgent),
    "hybrid": ("Hybrid", HybridFlightAgent),
    "mock": ("Mock-Agent (Giả lập độc lập)", MockFlightAgent),
}


def interactive_mode(model_name: str = "gemini-3.8-flash", no_api: bool = False):
    """Chế độ tương tác: chọn pattern, nhập ràng buộc."""

    if no_api or model_name.lower() in ("mock", "model_gia"):
        model_name = "mock"
        banner_extra = " [CHẾ ĐỘ NO-API / MOCK]"
    else:
        banner_extra = ""

    print("╔══════════════════════════════════════════════╗")
    print(f"║   🛫 AGENT ĐẶT VÉ MÁY BAY — SE373 BTVN#3{banner_extra:<5}║")
    print("╚══════════════════════════════════════════════╝")
    if model_name == "mock":
        print("⚡ Đang chạy chế độ NO-API: Sử dụng Model Giả Lập (Không cần API Key, không lo Rate Limit)\n")
    print()

    # Chọn pattern
    print("Chọn mẫu thiết kế agent:")
    print("  1. ReAct             (linh hoạt, từng bước)")
    print("  2. Plan-then-Execute (lên kế hoạch trước)")
    print("  3. Hybrid            (plan trước, ReAct khi fail)")
    print("  4. Mock-Agent        (agent giả lập chuyên biệt để test harness)")
    print()

    choice = ""
    while choice not in ("1", "2", "3", "4"):
        choice = input("Lựa chọn (1/2/3/4): ").strip()

    pattern_key = {"1": "react", "2": "plan", "3": "hybrid", "4": "mock"}[choice]
    pattern_name, AgentClass = PATTERNS[pattern_key]

    print(f"\n✅ Đã chọn: {pattern_name}")
    print()

    # Nhập ràng buộc
    print("─── Nhập ràng buộc (Enter để dùng mặc định) ───")

    def ask(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default

    tu = ask("Sân bay đi", "SGN")
    den = ask("Sân bay đến", "HAN")
    ngay = ask("Ngày bay (YYYY-MM-DD)", "2026-10-01")
    gio = ask("Bay trước giờ", "23:59")
    gia = ask("Giá tối đa (VND)", "2000000")
    so_ve = ask("Số vé", "1")
    ten = ask("Tên hành khách", "Nguyen Van A")

    constraints = RangBuoc(
        tu=tu.upper(), den=den.upper(), ngay=ngay,
        gio_muon_nhat=gio, gia_toi_da=int(gia.replace(",", "").replace(".", "")),
        so_ve=int(so_ve), ten_khach=ten,
    )

    print(f"\n📋 Ràng buộc: {constraints.to_prompt()}")
    print()

    # Chạy agent
    agent = AgentClass(
        constraints=constraints,
        model_name=model_name,
        verbose=True,
    )

    print(f"{'='*60}")
    print(f"🚀 Bắt đầu chạy {pattern_name}...")
    print(f"{'='*60}")

    result = agent.run(constraints.to_prompt())

    # In kết quả
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def single_pattern(pattern_key: str, model_name: str = "gemini-3.8-flash", no_api: bool = False):
    """Chạy 1 pattern với ràng buộc mặc định."""
    if no_api or model_name.lower() in ("mock", "model_gia"):
        model_name = "mock"
        print("⚡ [CHẾ ĐỘ NO-API] Sử dụng Model Giả Lập — Không tốn API Key & Tránh Rate Limit!\n")

    pattern_name, AgentClass = PATTERNS[pattern_key]

    constraints = RangBuoc(
        tu="SGN", den="HAN", ngay="2026-10-01",
        gio_muon_nhat="07:00", gia_toi_da=2_000_000,
        so_ve=1, ten_khach="Nguyen Van A",
    )

    print(f"🚀 Chạy {pattern_name} với ràng buộc: {constraints.to_prompt()}")
    print()

    agent = AgentClass(constraints=constraints, model_name=model_name, verbose=True)
    result = agent.run(constraints.to_prompt())

    print(f"\n{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Booking Agent — SE373 BTVN#3")
    parser.add_argument("--pattern", choices=["react", "plan", "hybrid", "mock"],
                        help="Chạy 1 pattern cụ thể (react, plan, hybrid, mock)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Chạy đánh giá 3 pattern")
    parser.add_argument("--no-api", action="store_true",
                        help="Dùng agent/model giả lập, không gọi API ngoài (tránh rate limit)")
    parser.add_argument("--model", default="gemini-3.8-flash",
                        help="Tên model Gemini (default: gemini-3.8-flash)")

    args = parser.parse_args()

    model = "mock" if args.no_api else args.model

    if args.evaluate:
        from evaluate import run_evaluation
        run_evaluation(model_name=model, no_api=args.no_api)
    elif args.pattern:
        single_pattern(args.pattern, model, no_api=args.no_api)
    else:
        interactive_mode(model, no_api=args.no_api)
