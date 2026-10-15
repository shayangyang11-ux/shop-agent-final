"""--dry-run 与测试共用的假 LLM 客户端：零费用验证评测管线。

与 conftest.fake_llm_client（队列脚本式）不同，这个假客户端是
"角色感知"的：按 system prompt 里的角色标记返回对应合同 JSON，
可以无脚本地跑完任意任务数 —— 评测管线（图驱动 + 指标 + judge +
落盘）全链路走真代码，只有模型是假的。
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace


def _page_name_from_user(user_text: str) -> str:
    m = re.search(r"页面：(.+?)（类型", user_text)
    return m.group(1) if m else "某页面"


def _reply(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_eval_fake_client():
    """返回 (假客户端, 调用记录列表)。"""

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        user = kwargs["messages"][-1]["content"]

        if "需求分析模块" in system:
            body = {
                "needs_summary": "评测假需求摘要",
                "product_categories": ["测试品类"],
                "recommended_entry": "one_click",
                "reason": "信息完整，适合一键生成",
                "missing_info": [],
            }
            return _reply(json.dumps(body, ensure_ascii=False))

        if "整店规划" in system:
            body = {
                "shop_name": "评测示例小店",
                "zones": [
                    {"name": "首页", "brief": "店铺门面"},
                    {"name": "商品专区", "brief": "商品陈列"},
                    {"name": "品牌故事", "brief": "信任背书"},
                ],
                "pages": [
                    {"name": "首页banner", "zone": "首页", "page_type": "banner",
                     "requirement_excerpt": "门面", "placeholder": False, "links": []},
                    {"name": "商品列表页", "zone": "商品专区", "page_type": "list",
                     "requirement_excerpt": "陈列", "placeholder": False,
                     "links": [{"trigger": "点击商品卡片", "to_page": "商品详情页", "label": "查看详情"}]},
                    {"name": "商品详情页", "zone": "商品专区", "page_type": "detail",
                     "requirement_excerpt": "转化", "placeholder": False,
                     "links": [{"trigger": "返回列表", "to_page": "商品列表页", "label": "返回"}]},
                    {"name": "品牌故事页", "zone": "品牌故事", "page_type": "about",
                     "requirement_excerpt": "背书", "placeholder": False,
                     "links": [{"trigger": "回到首页", "to_page": "首页banner", "label": "回首页"}]},
                ],
            }
            return _reply(json.dumps(body, ensure_ascii=False))

        # 注意分支顺序：judge 的 system 里也提到"页面文案"（防注入条款），
        # 必须先判 judge，再判页面设计师 —— 角色标记一律取最精确的短语。
        if "评测评分员" in system:
            body = {
                "coverage": 4, "copy_quality": 3, "consistency": 4, "conversion": 3,
                "rationale": "假裁决：结构完整、覆盖良好，文案略模板化。",
                "weak_points": ["组件文案偏通用"],
            }
            return _reply(json.dumps(body, ensure_ascii=False))

        if "页面文案设计师" in system:
            name = _page_name_from_user(user)
            body = {
                "headline": f"{name}：把好物讲清楚",
                "subheadline": "评测假文案：卖点具体、动线清晰",
                "components": [
                    {"type": "hero", "props": {"title": f"{name}主视觉", "cta": "立即选购"}},
                    {"type": "grid", "props": {"items": 6, "columns": 3}},
                ],
                "notes": "假客户端生成的评测占位内容",
            }
            return _reply(json.dumps(body, ensure_ascii=False))

        if "评测评分员" in system:
            body = {
                "coverage": 4, "copy_quality": 3, "consistency": 4, "conversion": 3,
                "rationale": "假裁决：结构完整、覆盖良好，文案略模板化。",
                "weak_points": ["组件文案偏通用"],
            }
            return _reply(json.dumps(body, ensure_ascii=False))

        # 兜底：不该有这次调用，返回一个必然校验失败的字符串让重试机制暴露问题
        return _reply('{"unexpected": true}')

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls
