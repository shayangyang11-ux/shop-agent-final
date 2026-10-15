"""generate_page 节点：Send 并行分支，单页内容生成与占位回填。

== 本节点的输入不是全局 state ==

它由 Send 激活，收到的 state 是 Send arg（{"draft": ..., "tree": ...}），
不是图的全局累积 state。fan_out_pages 里每个分支拿到的快照相同，
运行期间互不可见 —— 并行安全不靠锁，靠"不共享"。

== 返回值的归并路径 ==

返回 {"shop_tree": 部分树}，其中部分树只含：
- 该页的 zone（ShopTree 不变量要求页面分区存在）
- 该页本身：从快照中的 PLACEHOLDER 节点 model_copy(update={status,
  content}) 而来 —— 与批次1 冒烟测试 test_backfill_semantics 完全
  同构，这里就是它的全量形态。

LangGraph 把 N 个分支的返回值依次喂给 merge_tree reducer：
占位树 ⊕ 分支A ⊕ 分支B ... —— 规则2（占位+内容=回填）逐页生效，
分支完成顺序无关（交换律），重复投递无害（幂等）。

== 失败即占位（韧性设计的落点）==

structured_call 抛 LLMStructureError 时不写 error、不抛出：该页
在树里本来就是 PLACEHOLDER（分解节点落的），返回 {} 意味着
"这页留占位"。图继续、其他页照常生成 —— 单页失败不炸整图，
且占位状态天然支持后续重试回填。
"""

from __future__ import annotations

import threading

from shop_agent.llm import LLMServiceError, LLMStructureError, structured_call
from shop_agent.prompts import build_page_messages
from shop_agent.schema import (
    NodeStatus,
    PageContent,
    PageDraft,
    PageNode,
    ShopTree,
)
from shop_agent.state import ShopState

# ---------------------------------------------------------------------------
# 并发闸门：把 Send fan-out 的同时在飞的 LLM 请求压到 max_parallel_pages 以内
#
# LangGraph 用线程池跑 Send 分支：N 个分支 = N 个同时在飞的请求。
# 供应商限流（429）按"瞬时并发 + 分钟配额"计 —— 4 个请求齐射，免费档
# 账号几乎必撞。信号量让超出的分支排队等前面的完成，配合 llm.py 的
# 两层退避，构成"先防撞、撞了能自愈"的完整防御。
#
# 进程级单例：CLI 是单进程单图，全局一个闸门即正确；测试需要重置时
# 直接 monkeypatch 本模块的 _gate 为 None。
# ---------------------------------------------------------------------------

_gate: threading.Semaphore | None = None
_gate_lock = threading.Lock()


def _llm_gate() -> threading.Semaphore:
    """懒建信号量，容量取自配置 SHOP_AGENT_MAX_PARALLEL_PAGES（默认 3）。"""
    global _gate
    with _gate_lock:
        if _gate is None:
            from shop_agent.config import get_settings

            _gate = threading.Semaphore(max(1, get_settings().max_parallel_pages))
        return _gate


def generate_page(state: ShopState, client: object = None) -> dict:
    """并行分支体：为单个 draft 生成内容并构造可归并的部分树。"""
    # Send payload 是原生 dict（序列化边界友好的形态），此处验回强类型
    try:
        draft = PageDraft.model_validate(state["draft"])
        tree = ShopTree.model_validate(state["tree"])
    except (KeyError, ValueError):
        # payload 不合法 = 上游 bug（不是 LLM 问题），防御性返回空：
        # 宁可这页留占位，不让一个畸形任务打断其他 N-1 个分支
        return {}

    zone = tree.zones.get(draft.zone_id)
    if zone is None:  # 快照残缺，同上：防御性退化
        return {}

    siblings = [
        f"{p.name}（{p.page_type.value}）" for p in tree.pages.values() if p.id != draft.page_id
    ]

    try:
        with _llm_gate():  # 并发闸门：超出容量的分支在此排队
            content = structured_call(
                build_page_messages(
                    shop_name=tree.shop_name,
                    page_name=draft.name,
                    page_type=draft.page_type.value,
                    zone_name=zone.name,
                    requirement_excerpt=draft.requirement_excerpt,
                    siblings=siblings,
                ),
                PageContent,
                client=client,
            )
    except (LLMStructureError, LLMServiceError):
        # 失败即占位：结构错误（输出不合格）、限流（额度/频率不够）、
        # 超时/断连（服务端高峰变慢）都不炸图 —— 该页留 PLACEHOLDER，
        # 其他分支照常归并。一个分支把异常抛出去 = 整图崩溃、已完成
        # 分支的写入全部陪葬；退化成占位则可随时"补齐"续作。
        return {}

    # 占位回填：从快照取该页占位节点，只更新 status 与 content。
    # links / zone / name 等结构信息来自分解节点（规划事实），生成
    # 节点无权改动 —— 内容与结构的写权限分离。
    base = tree.pages.get(draft.page_id) or _minimal_placeholder(draft)
    filled = base.model_copy(update={"status": NodeStatus.DRAFT, "content": content})

    partial = ShopTree(
        shop_name="",  # 空店名：merge_tree 规则保证不覆盖已有店名
        zones={draft.zone_id: zone},
        pages={draft.page_id: filled},
    )
    return {"shop_tree": partial}


def _minimal_placeholder(draft: PageDraft) -> PageNode:
    """快照中找不到该页时的最小占位（防御路径，正常不应发生）。"""
    return PageNode(
        id=draft.page_id,
        zone_id=draft.zone_id,
        name=draft.name,
        page_type=draft.page_type,
        status=NodeStatus.PLACEHOLDER,
        content=None,
        links=[],
    )
