"""
agents/base.py — Lớp cơ sở cho các Agent
══════════════════════════════════════════
Chứa logic chung: tạo LLM (Gemini API hoặc Model giả lập), wrap tools thành LangChain tools, quản lý harness.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langchain_core.tools import StructuredTool

# Load .env
load_dotenv()

from lib.tool_call import ALL_TOOLS, TOOL_MAP, reset_db
from lib.harness import RangBuoc, Harness
from lib.model_gia import ModelGiaFlight


def get_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
    kich_ban: str = "tu_dong",
    **kwargs: Any,
):
    """Tạo ChatModel:
    - Nếu model_name là 'mock' / 'model_gia' hoặc USE_MOCK_MODEL=1:
      Trả về ModelGiaFlight chạy offline, tránh rate limit, test harness và tool calls.
    - Ngược lại: Kết nối Google Gemini API (ChatGoogleGenerativeAI).
      Nếu không có GOOGLE_API_KEY, tự động fallback sang ModelGiaFlight để tránh crash.
    """
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")

    # Kiểm tra cờ dùng model giả lập
    is_mock = (
        "mock" in model_name.lower()
        or "model_gia" in model_name.lower()
        or "fake" in model_name.lower()
        or "giả lập" in model_name.lower()
        or os.getenv("USE_MOCK_MODEL", "").lower() in ("true", "1", "yes")
    )

    if is_mock:
        return ModelGiaFlight(kich_ban=kich_ban, **kwargs)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "⚠️ [get_llm] Không tìm thấy GOOGLE_API_KEY trong .env! "
            "Tự động chuyển sang Model Giả Lập (ModelGiaFlight) để tiếp tục kiểm thử."
        )
        return ModelGiaFlight(kich_ban=kich_ban, **kwargs)

    from langchain_google_genai import ChatGoogleGenerativeAI

    primary_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=3,
        convert_system_message_to_human=True,
    )

    fallback_model = "gemini-2.5-flash" if model_name != "gemini-2.5-flash" else "gemini-1.5-flash"
    fallback_llm = ChatGoogleGenerativeAI(
        model=fallback_model,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=2,
        convert_system_message_to_human=True,
    )

    return primary_llm.with_fallbacks([fallback_llm])


def make_langchain_tools() -> list[StructuredTool]:
    """Wrap các tool Python thành LangChain StructuredTool.

    LangChain cần mỗi tool có:
    - name:        tên tool (agent gọi bằng tên này)
    - description: mô tả (agent đọc để quyết định gọi tool nào)
    - func:        hàm Python thật sự chạy
    """
    tools = []
    for fn in ALL_TOOLS:
        t = StructuredTool.from_function(
            func=fn,
            name=fn.__name__,
            description=fn.__doc__ or "",
        )
        tools.append(t)
    return tools


class BaseFlightAgent(ABC):
    """Lớp cơ sở trừu tượng cho Flight Booking Agent.

    Subclass cần implement:
        - run(user_request: str) -> dict
    """

    pattern_name: str = "base"   # override in subclass

    def __init__(
        self,
        constraints: RangBuoc,
        model_name: str | None = None,
        max_steps: int = 15,
        verbose: bool = True,
        kich_ban: str = "tu_dong",
        llm: Any = None,
    ):
        self.constraints = constraints
        self.harness = Harness(constraints)
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        self.max_steps = max_steps
        self.verbose = verbose
        self.kich_ban = kich_ban
        self.llm = llm or get_llm(self.model_name, kich_ban=self.kich_ban)
        self.tools = make_langchain_tools()

        # Metrics
        self.total_llm_calls = 0
        self.total_tool_calls = 0
        self.start_time = 0.0
        self.end_time = 0.0

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{self.pattern_name}] {msg}")

    def reset(self):
        """Reset agent state cho lần chạy mới."""
        reset_db()
        self.harness.reset()
        self.total_llm_calls = 0
        self.total_tool_calls = 0

    @abstractmethod
    def run(self, user_request: str) -> dict:
        """Chạy agent và trả về kết quả.

        Returns:
            {
                "ket_qua": "THANH_CONG" | "THAT_BAI",
                "booking": {...} | None,
                "ban_giao": {...} | None,
                "trace": "...",
                "metrics": {
                    "pattern": "...",
                    "llm_calls": int,
                    "tool_calls": int,
                    "time_seconds": float,
                },
            }
        """
        pass

    def _build_result(self) -> dict:
        """Xây dựng kết quả cuối cùng từ harness."""
        self.end_time = time.time()
        result = self.harness.get_result()
        result["trace"] = self.harness.get_trace()
        result["metrics"] = {
            "pattern": self.pattern_name,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "time_seconds": round(self.end_time - self.start_time, 2),
        }

        # Kiểm căn cứ nếu có câu trả lời
        if hasattr(self, "_final_answer") and self._final_answer:
            grounding = self.harness.check_grounding(self._final_answer)
            result["grounding"] = grounding

        return result

    def _execute_tool_with_harness(self, tool_name: str, args: dict) -> dict:
        """Chạy 1 tool qua harness: kiểm quyền → chạy tool → ghi log.

        Returns:
            dict: kết quả tool, hoặc {"status": "denied", "reason": ...}
        """
        self.total_tool_calls += 1

        # Kiểm phát hiện lặp
        loop_warning = self.harness.loop_detector.check(tool_name, args)
        if loop_warning:
            result = {"status": "error", "error": "loop_detected", "reason": loop_warning}
            self.harness.log_call(tool_name, args, result)
            self._log(f"⚠️  LOOP: {loop_warning}")
            return result

        # Kiểm quyền
        reason = self.harness.check_permission(tool_name, args)
        if reason:
            result = {"status": "denied", "reason": reason}
            self.harness.log_call(tool_name, args, result)
            self._log(f"🚫 DENIED: {tool_name}({args}) → {reason}")
            return result

        # Chạy tool thật
        try:
            fn = TOOL_MAP.get(tool_name)
            if fn is None:
                result = {"status": "error", "error": f"Tool '{tool_name}' không tồn tại."}
            else:
                result = fn(**args)
        except Exception as e:
            result = {"status": "error", "error": str(e)}

        self.harness.log_call(tool_name, args, result)
        self._log(f"🔧 {tool_name}({json.dumps(args, ensure_ascii=False)}) → {result.get('status', '?')}")
        return result


class MockFlightAgent(BaseFlightAgent):
    """Agent giả lập độc lập chuyên dùng để kiểm thử Harness, Tool Calls và App UI.

    Không cần kết nối internet hay API key. Cho phép chọn nhiều kịch bản test:
    - 'tu_dong': Tự động phân tích và xử lý luồng đặt vé tối ưu
    - 'lap': Kiểm thử cơ chế phát hiện lặp (LoopDetector)
    - 'vi_pham_quyen': Kiểm thử cơ chế chặn vi phạm ràng buộc (Permission Check)
    - 'bip_grounding': Kiểm thử phát hiện dữ kiện không rõ nguồn (Grounding Check)
    """

    pattern_name = "Mock-Agent"

    def __init__(self, constraints: RangBuoc, kich_ban: str = "tu_dong", **kwargs):
        super().__init__(
            constraints=constraints,
            model_name="mock",
            kich_ban=kich_ban,
            **kwargs,
        )

    def run(self, user_request: str) -> dict:
        self.reset()
        self.start_time = time.time()
        self._final_answer = ""

        llm_with_tools = self.llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content="Bạn là agent giả lập đặt vé máy bay dùng để test Harness."),
            HumanMessage(content=user_request),
        ]

        self._log(f"📝 [Mock Test] User: {user_request}")

        for step in range(self.max_steps):
            self.total_llm_calls += 1
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Nếu model không gọi tool nào → xong
            if not response.tool_calls:
                self._final_answer = response.content
                self._log(f"💬 [Mock Test] Model trả lời: {response.content[:150]}...")
                break

            # Thực thi tool qua harness
            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]
                tc_id = tc["id"]

                result = self._execute_tool_with_harness(tool_name, args)

                tool_msg = ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc_id,
                    name=tool_name,
                )
                messages.append(tool_msg)

                # Nếu bị harness chặn do lặp hoặc vi phạm nghiêm trọng
                if result.get("error") == "loop_detected":
                    self._log("⛔ [Mock Test] Dừng vòng lặp vì LoopDetector phát hiện lặp.")
                    return self._build_result()

        return self._build_result()
