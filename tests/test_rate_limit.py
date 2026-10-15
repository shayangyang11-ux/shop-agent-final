"""供应商侧故障防御测试：节流 / 退避 / 并发闸门 / 退化占位，零网络零费用。

背景：真实运行时"补齐 4 个占位页"先撞 429（速率限制），治好后又在
服务端高峰期撞上读超时 —— 两者都属于"服务端瞬时故障"，且都曾因
一个分支耗尽而炸掉整图。本文件锁死全部防御行为：
  1. llm.py 的请求节流 —— 拉开请求启动间距，防 RPM 类限流于未然；
  2. llm.py 的长退避 —— 429/超时后等 20s/40s 再试，耗尽抛应用层异常
     （LLMRateLimitError / LLMTimeoutError，openai 类型不出 llm.py）；
  3. generate_page 的信号量 —— 同时在飞的请求不超过 max_parallel_pages；
  4. 失败退化占位 —— 服务端故障耗尽不再炸图，该页留 PLACEHOLDER 可续作。
"""

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

import shop_agent.llm as llm
from shop_agent.config import get_settings
from shop_agent.llm import LLMRateLimitError, LLMTimeoutError, structured_call
from shop_agent.nodes.generate_page import generate_page
from shop_agent.schema import (
    InputAnalysis,
    NodeStatus,
    PageDraft,
    PageNode,
    PageType,
    ShopTree,
    ZoneNode,
)

VALID = (
    '{"needs_summary": "卖手工艺品的店", '
    '"recommended_entry": "one_click", '
    '"reason": "描述完整", '
    '"product_categories": ["手工艺品"]}'
)

_MESSAGES = [{"role": "user", "content": "分析需求"}]


def _rate_error() -> RateLimitError:
    """构造一个真实的 openai.RateLimitError（429）。"""
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("您的账户已达到速率限制", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    """构造一个真实的 openai.APITimeoutError（读超时）。"""
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return APITimeoutError(request=request)


def _ok_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ---------------------------------------------------------------------------
# 第一层防御：llm.py 的 429 长退避
# ---------------------------------------------------------------------------


def test_rate_limit_backs_off_then_succeeds(monkeypatch):
    """429 一次 -> 退避 20s -> 重试成功。退避被记录而非真等。"""
    sleeps: list[float] = []
    monkeypatch.setattr("shop_agent.llm._sleep", sleeps.append)

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _rate_error()
        return _ok_response(VALID)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = structured_call(_MESSAGES, InputAnalysis, client=client, rate_retries=2)

    assert result.needs_summary == "卖手工艺品的店"
    assert len(calls) == 2  # 失败 1 次 + 成功 1 次
    assert sleeps == [20.0]  # 第一次退避恰为基础值


def test_rate_limit_exhausts_and_raises(monkeypatch):
    """持续 429 -> 退避 20s、40s 后仍失败 -> LLMRateLimitError（带自救指引）。

    抛应用层异常而非裸的 RateLimitError：openai 的类型不越过 llm.py
    边界，换供应商时上层节点零改动。
    """
    sleeps: list[float] = []
    monkeypatch.setattr("shop_agent.llm._sleep", sleeps.append)

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise _rate_error()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(LLMRateLimitError) as exc_info:
        structured_call(_MESSAGES, InputAnalysis, client=client, rate_retries=2)

    assert len(calls) == 3  # 原始 1 次 + 退避重试 2 次
    assert sleeps == [20.0, 40.0]  # 指数退避
    assert isinstance(exc_info.value.__cause__, RateLimitError)  # 原始 429 链在异常上
    assert "SHOP_AGENT_MAX_PARALLEL_PAGES" in str(exc_info.value)  # 错误信息带自救指引


# ---------------------------------------------------------------------------
# 第二层防御：generate_page 的并发闸门
# ---------------------------------------------------------------------------


def _draft(i: int) -> PageDraft:
    return PageDraft(
        page_id=f"p-{i}",
        zone_id="z-home",
        name=f"测试页 {i}",
        page_type=PageType.LIST,
        requirement_excerpt="测试需求",
    )


def _payload(i: int) -> dict:
    """单个 Send 分支的 payload：draft + 树快照。"""
    tree = ShopTree(
        shop_name="闸门测试店",
        zones={"z-home": ZoneNode(id="z-home", name="首页")},
        pages={
            f"p-{j}": PageNode(
                id=f"p-{j}",
                zone_id="z-home",
                name=f"测试页 {j}",
                page_type=PageType.LIST,
                status=NodeStatus.PLACEHOLDER,
            )
            for j in range(4)
        },
    )
    return {"draft": _draft(i).model_dump(), "tree": tree.model_dump()}


def test_generate_page_caps_concurrency(monkeypatch):
    """4 个分支并发跑，同时在飞的 LLM 调用峰值 ≤ max_parallel_pages=2。

    shop_agent.nodes.__init__ 把函数 generate_page 重导出，遮蔽了同名
    子模块 —— 所以这里用 importlib 拿模块对象再打补丁（字符串路径
    会被解析到函数身上）。
    """
    import importlib

    monkeypatch.setenv("SHOP_AGENT_MAX_PARALLEL_PAGES", "2")
    get_settings.cache_clear()
    gpm = importlib.import_module("shop_agent.nodes.generate_page")
    monkeypatch.setattr(gpm, "_gate", None)

    lock = threading.Lock()
    stats = {"now": 0, "peak": 0}

    def create(**kwargs):
        with lock:
            stats["now"] += 1
            stats["peak"] = max(stats["peak"], stats["now"])
        time.sleep(0.05)  # 让并发窗口真实存在
        with lock:
            stats["now"] -= 1
        return _ok_response('{"headline": "测试标题"}')

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    try:
        threads = [
            threading.Thread(target=generate_page, args=(_payload(i), client)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats["peak"] <= 2, f"并发峰值 {stats['peak']} 超过了闸门容量 2"
        assert stats["peak"] == 2  # 容量没被浪费：确实同时跑到了 2
    finally:
        get_settings.cache_clear()  # 清掉带 env 覆盖的缓存，不污染后续测试


# ---------------------------------------------------------------------------
# 第三/四层防御：请求节流 + 限流退化占位（不炸图）
# ---------------------------------------------------------------------------


def test_pacing_spaces_request_starts(monkeypatch):
    """请求节流：间隔小于配置值时，第二次请求启动前被拉齐等待。

    防 RPM 类限流的核心：不是"同时在飞多少"（闸门管），而是
    "每分钟发出多少"（节流管）。
    """
    monkeypatch.setenv("SHOP_AGENT_LLM_MIN_REQUEST_INTERVAL", "0.25")
    get_settings.cache_clear()
    monkeypatch.setattr(llm, "_last_request_at", 0.0)

    sleeps: list[float] = []
    monkeypatch.setattr(llm, "_sleep", lambda s: sleeps.append(s))

    def create(**kwargs):
        return _ok_response(VALID)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    structured_call(_MESSAGES, InputAnalysis, client=client)  # 首发：无需等待
    structured_call(_MESSAGES, InputAnalysis, client=client)  # 二发：被节流拉齐

    assert len(sleeps) == 1  # 恰好一次节流等待（首发不等）
    assert 0 < sleeps[0] <= 0.25  # 等待量 = 间距配置减去已流逝时间


def test_generate_page_fails_soft_on_rate_limit(monkeypatch):
    """限流退避耗尽 -> 该页退化为占位（返回 {}），不炸图、不陪葬其他分支。

    这就是"补齐 4 页时 1 页撞限流"场景的正确结局：其余 3 页照常
    归并进树，限流页留 PLACEHOLDER，稍后再说一次「补齐」即可续作。
    """
    monkeypatch.setattr("shop_agent.llm._sleep", lambda s: None)  # 退避不真等

    def create(**kwargs):
        raise _rate_error()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = generate_page(_payload(0), client)

    assert result == {}  # 失败即占位：异常没有越过节点边界


# ---------------------------------------------------------------------------
# 同族故障：超时/断连（与限流共用退避与退化机制，LLMServiceError 基类）
# ---------------------------------------------------------------------------


def test_timeout_exhausts_and_raises(monkeypatch):
    """持续超时 -> 退避 20s、40s 后仍失败 -> LLMTimeoutError（带提速指引）。

    与限流同构但指引不同：限流教用户拉大间距，超时教用户等高峰过去
    或换更快的模型 —— 异常消息即排障手册。
    """
    sleeps: list[float] = []
    monkeypatch.setattr("shop_agent.llm._sleep", sleeps.append)

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise _timeout_error()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(LLMTimeoutError) as exc_info:
        structured_call(_MESSAGES, InputAnalysis, client=client, rate_retries=2)

    assert len(calls) == 3  # 原始 1 次 + 退避重试 2 次
    assert sleeps == [20.0, 40.0]  # 与限流同一套指数退避
    assert isinstance(exc_info.value.__cause__, APITimeoutError)  # 原始超时链在异常上
    assert "SHOP_AGENT_LLM_MODEL" in str(exc_info.value)  # 错误信息带提速指引


def test_generate_page_fails_soft_on_timeout(monkeypatch):
    """超时退避耗尽 -> 该页退化为占位（返回 {}），不炸图。

    真实场景：服务端高峰期单页生成超过超时阈值。结局与其他服务端
    故障一致 —— 图走完、报告列出未完成页、稍后「补齐」续作。
    """
    monkeypatch.setattr("shop_agent.llm._sleep", lambda s: None)  # 退避不真等

    def create(**kwargs):
        raise _timeout_error()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert generate_page(_payload(0), client) == {}  # 失败即占位
