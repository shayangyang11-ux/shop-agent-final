"""structured_call 的校验回填重试测试：假客户端穷举，零网络零费用。

工业级的意义：重试逻辑是本系统对 LLM 不确定性的核心防御，它自己
必须是确定性的、被测试锁死的。这些测试证明"错误回填"真的发生、
围栏真的被剥、耗尽真的会抛。
"""

from types import SimpleNamespace

import pytest

from shop_agent.config import get_settings
from shop_agent.llm import LLMStructureError, structured_call
from shop_agent.schema import InputAnalysis

VALID = (
    '{"needs_summary": "卖手工艺品的店", '
    '"recommended_entry": "one_click", '
    '"reason": "描述完整", '
    '"product_categories": ["手工艺品"]}'
)
SCHEMA_VIOLATION = '{"needs_summary": "x", "recommended_entry": "banana", "reason": "y"}'
NOT_JSON = "我觉得直接帮你装修好就行了"


def _fake_client(responses: list[str]):
    """最小可用的 OpenAI 兼容假客户端：按脚本顺序吐响应，记录每次调用。"""
    calls: list[dict] = []
    queue = list(responses)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=queue.pop(0)))]
        )

    client = SimpleNamespace()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    client.calls = calls
    return client


def _messages():
    return [
        {"role": "system", "content": "你是店铺需求分析器，输出 JSON。"},
        {"role": "user", "content": "帮我开一家咖啡店"},
    ]


def test_first_try_valid():
    client = _fake_client([VALID])
    result = structured_call(_messages(), InputAnalysis, client=client)

    assert isinstance(result, InputAnalysis)
    assert result.recommended_entry == "one_click"
    assert len(client.calls) == 1
    # 请求参数按配置注入
    s = get_settings()
    assert client.calls[0]["model"] == s.llm_model
    assert client.calls[0]["temperature"] == s.llm_temperature
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_retry_feeds_error_back():
    """语义层失败（枚举值非法）→ 原输出 + 校验错误回填进对话 → 第二次成功。"""
    client = _fake_client([SCHEMA_VIOLATION, VALID])
    result = structured_call(_messages(), InputAnalysis, client=client)

    assert isinstance(result, InputAnalysis)
    assert len(client.calls) == 2
    # 第二次请求的对话里包含：模型的错误原输出 + 含错误详情的修正指令
    retry_convo = client.calls[1]["messages"]
    assert retry_convo[-2]["role"] == "assistant"
    assert retry_convo[-2]["content"] == SCHEMA_VIOLATION
    assert retry_convo[-1]["role"] == "user"
    assert "recommended_entry" in retry_convo[-1]["content"]  # 错误细节真的在


def test_strips_markdown_fences():
    """json_mode 关闭时（供应商不支持 response_format），围栏仍被剥掉。"""
    fenced = f"```json\n{VALID}\n```"
    client = _fake_client([fenced])

    result = structured_call(_messages(), InputAnalysis, client=client, json_mode=False)

    assert isinstance(result, InputAnalysis)
    assert "response_format" not in client.calls[0]  # 确实未启用 json_object


def test_exhausts_retries_and_raises():
    """耗尽重试 → LLMStructureError，异常携带尝试次数与最后错误。"""
    client = _fake_client([NOT_JSON, NOT_JSON])

    with pytest.raises(LLMStructureError) as exc_info:
        structured_call(_messages(), InputAnalysis, client=client, retries=1)

    assert exc_info.value.attempts == 2
    assert exc_info.value.schema_name == "InputAnalysis"
    assert len(client.calls) == 2
