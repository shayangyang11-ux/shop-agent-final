"""共享测试基建：假 LLM 客户端工厂 + 全局旋钮隔离。

fake_llm_client（注入缝）：所有节点/图级测试共用 —— 按脚本顺序吐
响应、记录每次调用的 kwargs。队列耗尽即 IndexError：测试里不该
发生这次调用，发生了就让测试红掉。

_pinned_llm_knobs（旋钮隔离）：get_settings 读 .env 且带 lru_cache。
本机为 CLI 调的参数（如 SHOP_AGENT_LLM_MIN_REQUEST_INTERVAL=12）
若泄进测试：假客户端测试会真实睡 12 秒，sleeps 断言会混入节流等待。
所以每个测试前强制把请求节流归零、清掉设置缓存 —— CI 与本机都跑得
又快又稳。
"""

from types import SimpleNamespace

import pytest

from shop_agent.config import get_settings


@pytest.fixture
def fake_llm_client():
    """返回工厂：responses 脚本 -> (假客户端, 调用记录)。"""

    def _factory(responses: list[str]):
        calls: list[dict] = []
        queue = list(responses)

        def create(**kwargs):
            calls.append(kwargs)
            content = queue.pop(0)  # 队列空 = IndexError = 测试里不该有这次调用
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return client, calls

    return _factory


@pytest.fixture(autouse=True)
def _pinned_llm_knobs(monkeypatch):
    monkeypatch.setenv("SHOP_AGENT_LLM_MIN_REQUEST_INTERVAL", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
