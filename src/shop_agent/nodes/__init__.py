"""节点层：图的业务节点。

分层索引（面试讲架构的目录）：
- parse_input / choose_entry：需求解析 + 人机协同（批次3）
- planning.compile_plan：LLM 计划 -> 树 的确定性编译（批次4）
- modes：三入口分支节点 + Send fan-out 分发（批次4）
- generate_page：Send 并行分支，单页生成与占位回填（批次4）
- assemble：归并后收尾验收（批次4）
"""

from shop_agent.nodes.assemble import assemble, build_summary
from shop_agent.nodes.choose_entry import build_choice_payload, choose_entry, resolve_choice
from shop_agent.nodes.generate_page import generate_page
from shop_agent.nodes.modes import fan_out_pages, framework_mode, manual_mode, one_click_mode
from shop_agent.nodes.parse_input import parse_input
from shop_agent.nodes.planning import compile_plan

__all__ = [
    "assemble",
    "build_choice_payload",
    "build_summary",
    "choose_entry",
    "compile_plan",
    "fan_out_pages",
    "framework_mode",
    "generate_page",
    "manual_mode",
    "one_click_mode",
    "parse_input",
    "resolve_choice",
]
