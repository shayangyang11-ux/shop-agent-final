"""运行时状态层：LangGraph 的 ShopState 与归并 reducer。

一图流：节点返回 {"shop_tree": <ShopTree>} 的部分更新，LangGraph 不做
"后写覆盖"，而是对每个 key 调用其 Annotated reducer：

    新值 = reducer(当前值, 本次写入值)

本文件的核心是 merge_tree —— 它同时服务两个场景，且两个场景对它的
要求完全一致（幂等）：

场景 A：Send 并行 fan-out。N 个 generate_page 分支各自返回一棵
        "只含自己那页" 的部分树，LangGraph 依次调用
        merge_tree(累积树, 分支树) 完成归并。
        → 要求：交换律（分支完成顺序不定）、并行结果不互相清零。

场景 B：checkpointer 重放（你第 3 周实验的 at-least-once）。
        崩溃恢复后已落盘的节点输出会被重新应用，reducer 可能对
        "同样的数据" 被调用第二次。
        → 要求：幂等（merge(t, t) == t），否则恢复后状态损坏。

这就是为什么 reducer 不许写成 operator.add（序列拼接，A 场景树翻倍）
也不许用默认的 Last-Write-Wins（A 场景只剩最后一个分支的页面，
B 场景倒是幂等了但并行的意义没了）。

设计原则：本文件不 import langgraph —— 状态合同是纯类型层的，
框架耦合推迟到 graph.py（批次 5）。好处：reducer 可以脱离框架做
穷举测试，你眼前的 tests/test_state.py 就是在没有安装 langgraph
的环境里跑的。
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from shop_agent.schema import InputAnalysis, PageDraft, ShopTree


def merge_tree(current: Optional[ShopTree], update: Optional[ShopTree]) -> Optional[ShopTree]:
    """shop_tree 通道的归并 reducer：部分树 ⊕ 部分树 = 完整树。

    规则（按优先级）：
    1. update 为 None：无事发生，返回 current（节点没写这个 key 的旁路）。
    2. current 为 None：首写，直接采用 update。
    3. zones：同 id 分区，update 覆盖 current。
    4. pages 逐 id 归并：
       a. 新页面          → 直接收录（并行分支的 disjoint 页面在此汇合）；
       b. 占位 → 已填充    → 回填：新页覆盖占位（批次4 占位回填机制的归并形态）；
       c. 已填充 → 占位    → 拒绝降级：陈旧分支晚到，不许把已生成的
                             内容打回占位（这是真实系统里"乱序到达"
                             必须处理的脏情况）；
       d. 同为已填充      → update 胜。看似危险，实为幂等性的来源：
                             重放时 old/new 是同一份数据，覆盖等于不变。
    5. shop_name：非空者优先（update.shop_name or current.shop_name）。

    注意归并是"加法"不是"替换"：update 里没有、current 里有的页面会
    保留——框架模式整体重建树时，用户手动建过的页面不会被吞掉。
    刻意选择：宁可多留，不可误删。真要整体替换，用显式的清场动作，
    不在 reducer 里藏删除语义。

    实现是函数式的：不修改入参，返回新 ShopTree。reducer 的纯函数性
    是可测试性与幂等性的共同前提。
    """
    if update is None:
        return current
    if current is None:
        return update

    zones = {**current.zones, **update.zones}

    pages = dict(current.pages)
    for page_id, new_page in update.pages.items():
        old_page = pages.get(page_id)
        if old_page is None:
            pages[page_id] = new_page
        elif old_page.is_placeholder() and not new_page.is_placeholder():
            pages[page_id] = new_page  # 规则 4b：占位回填
        elif new_page.is_placeholder() and not old_page.is_placeholder():
            pages[page_id] = old_page  # 规则 4c：拒绝降级
        else:
            pages[page_id] = new_page  # 规则 4d：update 胜（幂等）

    return ShopTree(
        shop_name=update.shop_name or current.shop_name,
        zones=zones,
        pages=pages,
    )


class ShopState(TypedDict, total=False):
    """图的运行时状态。total=False：所有键可选，节点只返回自己写的键。

    键分两类，写代码前先想清楚每个键属于哪类：

    单写者键（无需 reducer，默认 Last-Write-Wins 即正确）：
        每个键在全图生命周期内只有一个节点会写它，不存在并发写。

    汇聚键（必须挂 reducer）：
        存在并行写入者（Send fan-out）或被重放（checkpointer），
        默认的覆盖语义会出错。写错不会在定义时报错，而是在运行中
        以"页面丢了/翻倍"的数据损坏形态出现——这是 Send 的第一坑。
    """

    # ---- 单写者键 ----
    user_input: str  # 用户原始输入（CLI 注入，无人再写）
    analysis: Optional[InputAnalysis]  # parse_input 节点写入
    entry_choice: Optional[str]  # choose_entry 节点写入（interrupt 恢复后的用户选择）
    page_drafts: Optional[list[PageDraft]]  # 分解节点写入的生成计划（Send 的数据源）
    error: Optional[str]  # 异常路径写入（异常输入用例走这里）
    summary: Optional[str]  # 收尾节点写入的执行报告（批次4：assemble / 各模式收口）

    # ---- 汇聚键 ----
    shop_tree: Annotated[Optional[ShopTree], merge_tree]  # 唯一的汇聚键：并行 + 重放
