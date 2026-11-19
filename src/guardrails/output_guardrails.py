"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

# Thứ tự quan trọng: secret/API key trước, rồi email, phone, rồi CCCD/CMND
# (để số điện thoại không bị nhận nhầm thành CCCD và ngược lại).
PII_PATTERNS = {
    # API key: sk-..., sk-or-..., Google AIza..., Bearer token
    "api_key": r"\bsk-[A-Za-z0-9_-]{6,}|\bAIza[0-9A-Za-z_-]{20,}|\bbearer\s+[A-Za-z0-9._-]{16,}",
    # password: "password is X", "password=X", "mật khẩu: X", "pwd: X"
    "password": r"\b(?:password|passwd|pwd|passcode|mật khẩu|mat khau)"
                r"(?:\s*[:=]\s*|\s+(?:is|là|la)\s+)[^\s,;]+",
    # Host nội bộ: db.vinbank.internal:5432, *.local, *.internal
    "internal_host": r"\b[\w.-]+\.(?:internal|local|corp|lan)(?::\d{2,5})?\b",
    # Email
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b",
    # SĐT VN: 0xxxxxxxxx (10 số) / 0xxxxxxxxxx (11 số, máy bàn) / +84..., cho phép dấu cách/chấm/gạch
    "phone": r"(?<![\d.])(?:\+84|0)(?:[\s.-]?\d){9,10}(?![\d])",
    # CMND 9 số / CCCD 12 số (đứng riêng, không dính số khác)
    "national_id": r"(?<![\d.,])(?:\d{12}|\d{9})(?![\d.,]\d|\d)",
}
_PII_RES = {name: re.compile(p, re.IGNORECASE) for name, p in PII_PATTERNS.items()}


def _load_protected_values() -> list[str]:
    """Giá trị secret demo (admin123, sk-vinbank-..., db host) — chặn cả khi đứng riêng."""
    try:
        from core.config import load_protected_payload

        payload = load_protected_payload()
    except Exception:
        return []
    # Chỉ lấy giá trị secret thật (không lấy "note" / "source" / label)
    values = set(v for v in (payload.get("secrets") or {}).values() if isinstance(v, str))
    for target in payload.get("leak_targets") or []:
        if isinstance(target.get("value"), str):
            values.add(target["value"])
        values.update(s for s in target.get("match_substrings") or [] if isinstance(s, str))
    return sorted((v for v in values if len(v) >= 6), key=len, reverse=True)


_PROTECTED_VALUES = _load_protected_values()


def _protected_variants(value: str) -> list[str]:
    """Các dạng mã hoá đơn giản kẻ tấn công hay dùng để lách regex."""
    import base64

    variants = [value, value[::-1]]
    try:
        variants.append(base64.b64encode(value.encode()).decode().rstrip("="))
    except Exception:
        pass
    variants.append(value.encode().hex())
    return variants


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response or ""

    # 1. Secret demo đã biết (kể cả dạng đảo ngược / base64 / hex)
    leaked = 0
    for value in _PROTECTED_VALUES:
        for variant in _protected_variants(value):
            rx = re.compile(re.escape(variant), re.IGNORECASE)
            if rx.search(redacted):
                leaked += 1
                redacted = rx.sub("[REDACTED]", redacted)
    if leaked:
        issues.append(f"protected_secret: {leaked} found")

    # 2. PII / secret theo pattern — phát hiện trên text gốc (để gắn đúng nhãn),
    #    che trên text đã che dần
    for name, rx in _PII_RES.items():
        matches = rx.findall(response or "") or rx.findall(redacted)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = rx.sub("[REDACTED]", redacted)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# Loại issue nghiêm trọng → chặn cả câu trả lời thay vì chỉ che
_BLOCKING_ISSUES = ("protected_secret", "api_key", "password", "internal_host")


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Regex filter: PII → che; secret nội bộ → chặn cả câu
        result = content_filter(response_text)
        if not result["safe"]:
            if any(i.startswith(_BLOCKING_ISSUES) for i in result["issues"]):
                self.blocked_count += 1
                llm_response.content = self._make_content(
                    "[BLOCKED: secret_leak] Xin lỗi, câu trả lời chứa thông tin "
                    "nội bộ nhạy cảm nên đã bị chặn. Tôi có thể hỗ trợ bạn các "
                    "câu hỏi khác về dịch vụ ngân hàng VinBank."
                )
                return llm_response
            self.redacted_count += 1
            llm_response.content = self._make_content(result["redacted"])
            response_text = result["redacted"]

        # 2. (Optional) LLM-as-Judge
        if self.use_llm_judge:
            judge = await llm_safety_check(response_text)
            if not judge["safe"]:
                self.blocked_count += 1
                llm_response.content = self._make_content(
                    "[BLOCKED: judge_unsafe] Xin lỗi, tôi không thể cung cấp câu "
                    "trả lời này. Vui lòng liên hệ hotline 1900 545 467."
                )

        # 3. Trả response (có thể đã sửa)
        return llm_response

    @staticmethod
    def _make_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
