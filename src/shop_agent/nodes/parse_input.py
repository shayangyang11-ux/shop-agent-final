"""parse_input 节点：原始输入 -> InputAnalysis（结构化需求分析）。

职责分层（背下来，面试讲架构就这套话）：
- prompts.py 管"对模型说什么"（话术资产，集中审计）
- llm.py     管"怎么调模型、怎么校验重试"（基础设施）
- 本节点     管"编排与数据流"：读 state、调基座、写 state、转异常

节点级错误处理哲学：LLM 的不确定性在本层终结 —— 捕获 LLMStructureError
写入 state.error，让条件边走 abort 路径。图不裸崩、用户拿到人话错误，
这是"异常是数据不是事故"的落地。
"""

from __future__ import annotations

from shop_agent.llm import LLMServiceError, LLMStructureError, structured_call
from shop_agent.prompts import build_parse_messages
from shop_agent.schema import InputAnalysis
from shop_agent.state import ShopState

# 短于该长度的输入直接判异常，不浪费一次 LLM 调用 —— 在边界上用
# 便宜的手段挡掉明显无效的输入，贵的手段留给值得的输入。
MIN_INPUT_CHARS = 6


def parse_input(state: ShopState, client: object = None) -> dict:
    """解析用户输入为 InputAnalysis。

    client 参数是依赖注入缝（DI seam）：LangGraph 调用节点时只传
    state，client 保持 None 走真实客户端；测试直接注入假客户端，
    穷举成功 / 空输入 / 重试耗尽三条路径，零网络零费用。
    """
    raw = (state.get("user_input") or "").strip()

    if len(raw) < MIN_INPUT_CHARS:
        return {
            "error": (
                f"输入为空或过短（{len(raw)} 字）。"
                "请描述你的商品品类、风格偏好，或期望的页面。"
            )
        }

    try:
        analysis = structured_call(build_parse_messages(raw), InputAnalysis, client=client)
    except LLMServiceError as err:
        # 限流/超时不是输入的错，指引也与结构错误不同：等一等，而不是改描述
        return {"error": f"{err}（本次输入未消耗，稍后直接重试即可）"}
    except LLMStructureError as err:
        return {"error": f"需求解析失败：{err}。请换一种描述方式再试。"}

    return {"analysis": analysis}
