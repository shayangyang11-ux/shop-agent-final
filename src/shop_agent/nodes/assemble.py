"""assemble 节点：所有 generate_page 分支归并后的收尾（barrier 下游）。

它与 generate_page 之间是普通边 —— LangGraph 的 superstep 语义
保证 N 个并行分支全部完成、merge_tree 归并完毕后，assemble 才
执行一次。这就是免费的 barrier：汇聚是图拓扑的固有语义，不是
手写的计数器。

职责：对归并结果做终验与汇报。它不做任何生成 —— "验收"和"施工"
分离，验收逻辑才可信（施工者不自证完工）。

验收判据与批次1 的树合同一一对应：
- placeholder_page_ids()：生成失败 / 信息不足而留占位的页面
- dangling_links()：指向占位页的跳转（回填完成的判据是清零）
一个占位都没剩 = 全部回填成功。
"""

from __future__ import annotations

from shop_agent.state import ShopState


def build_summary(tree) -> str:
    """纯函数：树 -> 人话报告。CLI 与测试共用，展示逻辑不重复。"""
    total = len(tree.pages)
    placeholders = tree.placeholder_page_ids()
    dangling = tree.dangling_links()

    if not placeholders:
        return f"整店生成完成：{len(tree.zones)} 个分区、{total} 个页面全部回填，无悬空跳转。"

    lines = [f"部分完成：{total} 页中 {total - len(placeholders)} 页已生成，{len(placeholders)} 页留待回填："]
    name_by_id = {pid: tree.pages[pid].name for pid in placeholders}
    for pid in placeholders:
        lines.append(f"  - {name_by_id[pid]}（{pid}）")
    if dangling:
        lines.append(f"另有 {len(dangling)} 条跳转指向占位页，补齐后自动解除悬空。")
    lines.append("对这些页面说「补齐」即可继续（单页重试走同样的回填路径）。")
    return "\n".join(lines)


def assemble(state: ShopState) -> dict:
    """收尾节点：读归并后的树，写 summary。只读不写树 —— 验收不改数据。"""
    tree = state.get("shop_tree")
    if tree is None:
        return {"summary": "没有可汇总的店铺树（本次运行未产生结构）。"}
    return {"summary": build_summary(tree)}
