"""choose_entry 节点：三入口人机协同的 interrupt。

== 本项目最重要的一课：interrupt 不是"暂停点"，是"节点重跑" ==

LangGraph 的 interrupt() 实际抛出 GraphInterrupt 异常，图捕获后把
「上一个已完成节点（parse_input）的状态」写入 checkpointer。恢复时
（graph.invoke(Command(resume=...))），choose_entry 整个节点从头
重新执行 —— interrupt() 这一行第二次执行才返回 resume 携带的值。

由此推出两条设计铁律：

推论一：interrupt() 之前的所有计算，恢复时全部白算。
        所以节点体只有三件事：读已落盘的 state、interrupt、写结果。
        没有 LLM 调用、没有文件写入 —— 重跑零成本、零重复副作用。

推论二（原大作业批改修正的落地）：interrupt 必须独立成小节点。
        若把"解析输入 + interrupt"挤在一个节点，恢复时那次 LLM 解析
        会被重复执行 —— 钱和不确定性都翻倍。需求解析在 parse_input
        完成并落盘，choose_entry 只消费分析结果。

== 菜单的分层 ==

build_choice_payload 决定"展示什么"（数据），渲染层决定"怎么展示"
（终端 / Web）。Agent 不关心前端形态 —— 这就是同一张图可以先后接
CLI 和 Web UI 的原因。
"""

from __future__ import annotations

from typing import Any, Optional

from shop_agent.schema import EntryChoice, InputAnalysis
from shop_agent.state import ShopState

_ENTRY_OPTIONS: list[dict] = [
    {
        "value": EntryChoice.MANUAL.value,
        "label": "手动创建",
        "desc": "Agent 只响应指令，不主动生成（想完全掌控时选这个）",
    },
    {
        "value": EntryChoice.FRAMEWORK.value,
        "label": "生成框架",
        "desc": "生成整店结构与占位页面，内容留空后补",
    },
    {
        "value": EntryChoice.ONE_CLICK.value,
        "label": "AI 一键补齐",
        "desc": "结构 + 全部页面内容并行生成（需求完整时最快）",
    },
]

_VALID_CHOICES = {choice.value for choice in EntryChoice}


def build_choice_payload(analysis: Optional[InputAnalysis]) -> dict:
    """构造 interrupt 暂停时展示给用户的选择菜单。纯函数。"""
    if analysis is None:
        # 防御路径：路由保证 analysis 存在才进来，但边界函数不信任上游
        return {
            "title": "请选择店铺装修模式",
            "options": _ENTRY_OPTIONS,
            "recommended": EntryChoice.MANUAL.value,
            "reason": "（未获取到需求分析，建议先手动探索）",
            "missing_info": [],
        }
    return {
        "title": "请选择店铺装修模式",
        "options": _ENTRY_OPTIONS,
        "recommended": analysis.recommended_entry.value,
        "reason": analysis.reason,
        "missing_info": analysis.missing_info,
    }


def resolve_choice(raw: Any) -> str:
    """校验/归一化 resume 值。纯函数。

    无效输入回退 manual 而不是回退推荐值 —— fail-safe defaults：
    用户敲错字（或上游传了脏数据）时，"什么都不生成、等下一步指令"
    是唯一不会造成意外后果的选项。宁可不生成，不可乱生成。

    接受两种形态：字符串（"one_click"）或 {"entry": "one_click"}。
    后者留给未来的结构化前端 —— 协议宽容在边界，严格在内核。
    """
    if isinstance(raw, str) and raw in _VALID_CHOICES:
        return raw
    if isinstance(raw, dict):
        value = raw.get("entry")
        if isinstance(value, str) and value in _VALID_CHOICES:
            return value
    return EntryChoice.MANUAL.value


def choose_entry(state: ShopState) -> dict:
    """节点体：读 state -> interrupt -> 写选择。三行，零副作用。"""
    from langgraph.types import interrupt  # 延迟导入：框架边界压缩进这一行

    raw = interrupt(build_choice_payload(state.get("analysis")))
    return {"entry_choice": resolve_choice(raw)}
