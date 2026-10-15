"""规划编译层：TreePlan（LLM 输出）-> ShopTree + PageDraft 列表。

== 确定性收口（本文件存在的理由）==

LLM 负责意图（"要有首页、商品列表、三个详情页"），本文件负责机器
事实（id 分配、链接翻译、不变量维护）。两者边界清晰：

- LLM 不编机器 id：让 LLM 输出 "p-001" 这种 id 是可靠性灾难 ——
  id 必须由确定性代码分配，才谈得上"列表页的 link 指向详情页"这种
  精确引用。所以 PlannedPage 用名字互引，翻译在这里收口。

- 幻觉链接丢弃：LLM 引用一个不存在的页面名（幻觉）时，这条 link
  被丢弃并计数返回 —— 防御不靠 prompt 恳求，靠编译期校验。
  prompt 只能降低幻觉概率，编译层保证幻觉进不了数据结构。

== 为什么全部页面先落 PLACEHOLDER ==

"结构先行、内容回填"是数据模型层的韧性设计：
- 框架模式：compile 完即结束，树上全是占位 —— 合法终态；
- 一键模式：Send 分支的生成结果经 merge_tree 规则2（占位+内容=回填）
  逐页替换占位 —— 生成失败的页面自然留占位，整图不崩；
- 后续重试：用户对占位页说"补齐这页"，再次走回填路径即可。
占位不是"没做完"，是"任何时刻都合法的状态"。
"""

from __future__ import annotations

from shop_agent.schema import (
    LinkSpec,
    NodeStatus,
    PageDraft,
    PageNode,
    PlannedPage,
    ShopTree,
    TreePlan,
    ZoneNode,
)


def _allocate_ids(plan: TreePlan) -> tuple[dict[str, str], dict[str, str], list[PlannedPage]]:
    """给分区与页面分配稳定 id。

    稳定的含义：同一份 plan 每次编译得到相同 id（确定性），pages 按
    LLM 给出的顺序编号 —— 顺序即重要性，id 前缀 p-/z- 表达类别。

    页面重名：LLM 被要求名字唯一但不可信。防御 = 编译期改名
    （"详情页" -> "详情页#2"），保证 id 分配不塌缩。

    纯函数纪律：不改入参。重名页的改名结果通过第三个返回值传出，
    compile_plan 使用改名后的副本 —— 与 merge_tree 的不可变纪律
    一致（可测试性与幂等性的共同前提）。
    """
    zone_ids: dict[str, str] = {}
    for i, zone in enumerate(plan.zones, start=1):
        zone_ids[zone.name] = f"z-{i:03d}"

    page_ids: dict[str, str] = {}
    seen_names: set[str] = set()
    renamed: list[PlannedPage] = []
    for i, page in enumerate(plan.pages, start=1):
        name = page.name
        if name in seen_names:  # 重名防御：改名而不是合并
            suffix = 2
            while f"{page.name}#{suffix}" in seen_names:
                suffix += 1
            page = page.model_copy(update={"name": f"{page.name}#{suffix}"})
        renamed.append(page)
        seen_names.add(page.name)
        page_ids[page.name] = f"p-{i:03d}"
    return zone_ids, page_ids, renamed


def compile_plan(plan: TreePlan) -> tuple[ShopTree, list[PageDraft], int]:
    """编译规划为 (全占位树, 待生成任务列表, 被丢弃的幻觉链接数)。

    返回三元组而非对象：调用方（modes.py）各取所需，框架模式只要
    树，一键模式还要 drafts。丢弃计数进 summary，让"幻觉发生了"
    对用户可见而不是静默吞掉 —— 可观测性是防御的一部分。
    """
    zone_ids, page_ids, planned_pages = _allocate_ids(plan)

    # 未知分区的页面不丢弃（内容价值高于结构洁癖），归入默认分区
    DEFAULT_ZONE = "z-misc"

    zones: dict[str, ZoneNode] = {
        zone_ids[z.name]: ZoneNode(id=zone_ids[z.name], name=z.name) for z in plan.zones
    }

    pages: dict[str, PageNode] = {}
    drafts: list[PageDraft] = []
    dropped_links = 0

    for planned in planned_pages:
        zone_id = zone_ids.get(planned.zone)
        if zone_id is None:  # LLM 引用了未声明的分区 -> 兜底分区
            zone_id = DEFAULT_ZONE
            zones.setdefault(zone_id, ZoneNode(id=DEFAULT_ZONE, name="其他"))
        page_id = page_ids[planned.name]

        # 链接翻译：页面名 -> 页面 id；查无此名 = 幻觉，丢弃
        translated: list[LinkSpec] = []
        for link in planned.links:
            target = page_ids.get(link.to_page)
            if target is None:
                dropped_links += 1
                continue
            translated.append(link.model_copy(update={"to_page": target}))

        pages[page_id] = PageNode(
            id=page_id,
            zone_id=zone_id,
            name=planned.name,
            page_type=planned.page_type,
            status=NodeStatus.PLACEHOLDER,  # 结构先行：全部先落占位
            content=None,
            links=translated,
        )

        if not planned.placeholder:  # 非占位页才有生成任务
            drafts.append(
                PageDraft(
                    page_id=page_id,
                    zone_id=zone_id,
                    name=planned.name,
                    page_type=planned.page_type,
                    requirement_excerpt=planned.requirement_excerpt,
                )
            )

    tree = ShopTree(shop_name=plan.shop_name, zones=zones, pages=pages)
    return tree, drafts, dropped_links
