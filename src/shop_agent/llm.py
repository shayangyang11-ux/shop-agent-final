"""LLM 基座：OpenAI 兼容客户端 + 结构化输出（校验回填重试）。

两个工业化主题：

1. 结构化输出 ≠ json_object。
   让模型"输出 JSON"有两层失败：语法层（不是合法 JSON，如 markdown
   围栏、尾逗号、截断）和语义层（是 JSON 但不满足 Schema，如枚举值
   拼错、必填字段缺失）。response_format=json_object 只能兜语法层；
   语义层必须靠 Pydantic 校验 + 把校验错误喂回对话重试——也就是你
   手写 v1 里"JSON 解析容错 + 错误回填重试"的工业化版本。

2. 依赖注入而非全局连接。
   openai 延迟到 get_client() 内部才 import（模块导入零开销、可被
   测试替身替换）；structured_call 接受 client 参数，单测用假客户端
   穷举重试路径，不花一分钱 API 费。测试不依赖网络，才敢放进 CI。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional, TypeVar, Union

from pydantic import BaseModel, ValidationError

from shop_agent.config import get_settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

def _sleep(seconds: float) -> None:
    """间接一层，让测试能 monkeypatch 掉真实的等待（不让 CI 睡 40 秒）。"""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# 请求节流（pacing）：限制"请求启动频率"，而不只是"同时在飞的数量"
#
# 为什么有了 generate_page 的并发闸门还要节流：闸门治的是瞬时齐射，
# 但供应商更常见的限流维度是 RPM（每分钟请求数）。账号档位 RPM=5 时，
# 就算完全串行，不隔间距地连发 5 次也照样撞墙。节流把分钟窗口内的
# 请求预算摊平 —— 防撞永远比自愈便宜。
# ---------------------------------------------------------------------------

_pace_lock = threading.Lock()
_last_request_at = 0.0  # 上次请求启动时刻的 time.monotonic() 时间戳


def _pace(min_interval: float) -> None:
    """保证本次请求与上一次请求的启动间隔不少于 min_interval 秒。"""
    global _last_request_at
    if min_interval <= 0:
        return
    with _pace_lock:
        now = time.monotonic()
        wait = _last_request_at + min_interval - now
        if wait > 0:
            _sleep(wait)
            now = time.monotonic()
        _last_request_at = now

# 重试时追加给模型的两条消息的模板
_RETRY_USER_TMPL = (
    "你上一次的输出未通过校验，错误如下：\n{err}\n\n"
    "请严格修正后重新输出。要求：只输出一个 JSON 对象，不要包含任何"
    "解释、markdown 代码围栏或其他文本。"
)


class LLMStructureError(RuntimeError):
    """结构化输出耗尽重试次数仍不合法。携带最后一次错误供上层路由。"""

    def __init__(self, schema_name: str, attempts: int, last_error: Union[json.JSONDecodeError, ValidationError]):
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM 输出在 {attempts} 次尝试后仍未通过 {schema_name} 校验：{last_error!s:.200}"
        )


class LLMServiceError(RuntimeError):
    """服务端瞬时故障基类：限流（429）/ 超时 / 断连。

    共同语义：不是输入或提示词的错，等一等再试有意义；但退避耗尽后
    调用方应"退化"而非崩溃 —— generate_page 留占位（图不崩、已完成
    分支不陪葬），解析/规划节点写 state.error（走 abort 路由给人话）。
    openai 的异常类型不越过 llm.py 边界：换供应商时上层零改动。
    """


class LLMRateLimitError(LLMServiceError):
    """429 退避耗尽：本层等完了、供应商仍在拒绝（账号额度/频率不够）。"""


class LLMTimeoutError(LLMServiceError):
    """超时/断连，退避重试后仍失败（长任务遇服务端高峰期变慢）。"""


def get_client():
    """进程级懒加载的 OpenAI 兼容客户端。

    延迟 import 的两个理由：
    - import openai 有可感知的冷启动开销，CLI 里不调 LLM 的子命令
      （如跑测试、渲染预览）不该为此买单；
    - 测试可以用 monkeypatch 替换本函数返回假客户端。
    """
    from openai import OpenAI  # 延迟导入

    settings = get_settings()
    if not settings.is_configured:
        raise RuntimeError(
            "未配置 API Key：请复制 .env.example 为 .env 并填写 SHOP_AGENT_LLM_API_KEY"
        )
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _create_with_rate_backoff(
    client: object,
    kwargs: dict,
    retries: int,
    min_interval: float,
    backoff_base: float,
):
    """带请求节流 + 瞬时故障长退避的请求层（SDK 自带重试之上再兜一层）。

    防御分四段：
    - 节流（pacing）：拉开请求启动间距，防 RPM 类限流于未然；
    - SDK 重试（max_retries）：秒级指数退避，抹平瞬时抖动与偶发超时；
    - 本层长退避（backoff_base 起指数递增）：429 等配额窗口翻页，
      超时/断连等故障给服务端喘息后再试；
    - 耗尽转译：抛应用层异常（LLMRateLimitError / LLMTimeoutError），
      openai 的类型不越过 llm.py 的边界（换供应商时上层零改动）。
    """
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:  # pragma: no cover - 无 openai 的纯替身环境
        RateLimitError = APITimeoutError = APIConnectionError = ()

    def _exhausted_wait() -> float:
        return sum(backoff_base * (2**i) for i in range(retries))

    for attempt in range(retries + 1):
        _pace(min_interval)
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as err:
            if attempt >= retries:
                raise LLMRateLimitError(
                    f"LLM 供应商持续限流（429）：退避共 {_exhausted_wait():.0f} 秒后仍被拒绝。"
                    "请稍等几分钟再试；仍频繁撞限时，可在 .env 中设置 "
                    "SHOP_AGENT_MAX_PARALLEL_PAGES=1（完全串行）与 "
                    "SHOP_AGENT_LLM_MIN_REQUEST_INTERVAL=12（拉大请求间距）。"
                ) from err
            _backoff_wait(attempt, retries, backoff_base, kind="供应商限流（429）")
        except (APITimeoutError, APIConnectionError) as err:
            if attempt >= retries:
                raise LLMTimeoutError(
                    f"LLM 请求超时/连接失败：单次超时 {kwargs.get('timeout')} 秒，"
                    f"退避共 {_exhausted_wait():.0f} 秒后重试仍失败。"
                    "页面生成是长任务（最多 4096 tokens），服务端高峰期会变慢 —— "
                    "稍后重试即可；追求速度可把 .env 的 SHOP_AGENT_LLM_MODEL "
                    "换成 glm-4.5-flash 等更快的型号。"
                ) from err
            _backoff_wait(attempt, retries, backoff_base, kind="请求超时/断连")


def _backoff_wait(attempt: int, retries: int, backoff_base: float, *, kind: str) -> None:
    """退避等待，由外层循环重试；kind 仅用于日志区分故障家族。"""
    wait = backoff_base * (2**attempt)
    logger.warning(
        "LLM %s，退避 %.0f 秒后重试（第 %d/%d 次）……",
        kind,
        wait,
        attempt + 1,
        retries,
    )
    _sleep(wait)


def _extract_json(raw: str) -> str:
    """剥掉模型常见的 markdown 围栏（```json ... ```）。

    json_object 模式下通常不会出现围栏，但换供应商/换模型后就可能，
    兜一层比在异常里猜"为什么又挂了"便宜得多。
    """
    text = raw.strip()
    if text.startswith("```"):
        # 去掉首行 ```json / ```，去掉尾部 ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def structured_call(
    messages: list[dict],
    schema: type[T],
    *,
    client: Optional[object] = None,
    json_mode: bool = True,
    retries: Optional[int] = None,
    rate_retries: Optional[int] = None,
) -> T:
    """带校验回填重试的结构化调用：messages -> schema 实例。

    流程（共 retries+1 次机会）：
        请求(json_object) -> 剥围栏 -> json.loads -> schema.model_validate
        任一步失败：把【原输出 + 校验错误】追加进对话，让模型看着
        自己的错误改——错误信息本身就是最高质量的修复指令。

    参数：
        client:   注入的客户端（测试用）；缺省时懒加载真实客户端。
        json_mode: 是否启用 response_format=json_object。个别供应商
                  不支持该参数时置 False，靠提示词约束 + 剥围栏兜底。
        retries:  校验失败后的重试次数，缺省读配置。
        rate_retries: 429 限流后的长退避重试次数，缺省读配置。

    抛出：
        LLMStructureError —— 校验重试耗尽（模型输出不合格）。
        LLMRateLimitError / LLMTimeoutError —— 服务端故障退避耗尽
        （限流/超时/断连）。调用方（节点）应捕获后退化处理：写
        state.error 或留占位，而不是让图裸崩。
    """
    settings = get_settings()
    client = client if client is not None else get_client()
    attempts = (settings.llm_struct_retries if retries is None else retries) + 1
    rate_attempts = (
        settings.llm_rate_backoff_retries if rate_retries is None else rate_retries
    )

    convo = list(messages)
    last_error: Union[json.JSONDecodeError, ValidationError, None] = None

    for _ in range(attempts):
        kwargs: dict = {
            "model": settings.llm_model,
            "messages": convo,
            "temperature": settings.llm_temperature,
            "timeout": settings.llm_timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = _create_with_rate_backoff(
            client,
            kwargs,
            rate_attempts,
            settings.llm_min_request_interval,
            settings.llm_rate_backoff_base,
        )
        raw = response.choices[0].message.content or ""

        try:
            return schema.model_validate(json.loads(_extract_json(raw)))
        except (json.JSONDecodeError, ValidationError) as err:
            last_error = err
            # 错误回填：模型看得见自己的原输出与判错理由
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _RETRY_USER_TMPL.format(err=err)},
            ]

    raise LLMStructureError(schema.__name__, attempts, last_error)  # type: ignore[arg-type]
