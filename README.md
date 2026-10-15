# shop-agent · AI 电商店铺装修 Agent

> LangGraph 编排范式的完整工程实现：**interrupt 人机协同 · Send 并行 fan-out · 条件边路由 · checkpointer 持久化 · 占位节点回填**

## 这是什么

面向无编程能力商家的店铺装修 Agent：输入一段商品/风格描述，系统解析需求后给出三种装修模式（手动创建 / 生成框架 / AI 一键补齐），一键模式下按页面并行生成内容，并支持"先生成结构、后回填内容"的占位回填机制。

```
START → parse_input ─error→ END（人话错误，已落盘）
          │ok
          ▼
      choose_entry ←—— interrupt：渲染菜单，暂停等人
          │ Command(resume=...)
          ├─ manual_mode ──────────────────────────→ END（待命，不建树）
          ├─ framework_mode ───────────────────────→ END（全占位树）
          └─ one_click_mode（落全占位树 + drafts）
                │ fan_out_pages（条件边）
                ├─ Send ×N ─→ generate_page ─┐   ← 并行分支，互不可见
                └─ ["assemble"]（空计划兜底）─┤
                                             ▼   ← superstep barrier：全部完成后
                                         assemble（验收 + 报告）→ END

会话内补齐（对占位页再入）：
START → fan_out_placeholders → Send ×N → generate_page → assemble → END
```

## 快速开始

```bash
# 1. 安装（建议 conda 环境内）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest   # 跑测试用

# 2. 配置 API Key（OpenAI 兼容协议，默认智谱 GLM，可一行换供应商）
cp .env.example .env    # 填 SHOP_AGENT_LLM_API_KEY

# 3. 运行
python -m shop_agent.cli                # 正式 CLI（多轮会话 + 崩溃恢复 + 补齐）
python -m shop_agent.cli --resume <id>  # 崩溃后续接会话
python -m shop_agent.preview            # 最小图活文档（单轮三模式演示）

# 4. 测试（无需 API Key，LLM 全部为注入的假客户端）
python -m pytest
```

## 五个验收用例

| # | 用例 | 操作 | 预期 | 对应测试 |
|---|------|------|------|----------|
| 1 | 完整需求一键生成 | 输入完整店铺描述 → 选「一键补齐」 | 并行生成全部页面，无占位、无悬空跳转 | `test_acceptance.py::test_case1_full_requirement_one_click` |
| 2 | 模糊需求生成框架 | 输入简短方向性描述 → 选「生成框架」 | 只有树结构 + 占位页，零内容生成 | `test_acceptance.py::test_case2_vague_framework` |
| 3 | 未开始需求手动模式 | 输入问句式描述 → 选「手动创建」 | 不建树，Agent 待命 | `test_acceptance.py::test_case3_manual_standby` |
| 4 | 异常输入拦截 | 输入过短文本（<6 字） | 人话错误提示，图正常终止，零 LLM 调用 | `test_acceptance.py::test_case4_invalid_input_aborts` |
| 5 | 崩溃恢复 | 菜单处 Ctrl+C → `--resume` 同会话 | 从中断点恢复，需求解析不重跑（计费调用数不增） | `test_acceptance.py::test_case5_crash_recovery` |

## 评测体系：LLM-as-judge 最小闭环（eval/ 子包）

生成质量不能靠"看起来不错"——本仓库内置一条可复现的评测流水线：

```
20 个开店需求（多行业 × 难度梯度）
  → 生产图 one_click 真路径生成整店（评测与用户走同一条代码路径）
  → 确定性结构指标（占位数/悬空链接/组件数，代码算，不花 token）
  → LLM-as-judge 四维评审（需求覆盖/文案质量/整店一致性/转化设计，
    每维 1-5 分且写死 3/5 锚点，树渲染而非透传 JSON，防提示词注入）
  → annotation.csv 人工盲标（不含 judge 分）→ 一致率分析
```

```bash
python -m shop_agent.eval run --dry-run   # 零费用自检管线（角色感知假客户端）
python -m shop_agent.eval run             # 真跑：生成 + 评审，产物落 eval_results/
python -m shop_agent.eval agreement 标注.csv   # 人工标完算一致率
```

一致率三指标各答一个问题：**Pearson**（标尺是否一致）、**Spearman**（排序对不对——
决定能否用 judge 做"最差 20%"回归门禁）、**QWK**（有序评分的标准一致率，判读基准
QWK≥0.8 强一致）。分歧 ≥2 分的任务自动进人工复核清单。统计实现零科学计算栈依赖
（手写 Pearson/Spearman/加权 kappa，手算向量钉死正确性）。

## 架构

### 分层（框架只出现在边缘）

```
schema.py    领域模型（Pydantic）      ← 零框架依赖
state.py     ShopState + merge_tree    ← 零框架依赖（reducer 可脱离框架穷举测试）
config.py    环境配置                  ← 零框架依赖
llm.py       LLM 基座：结构化输出 + 校验回填重试（openai 延迟导入）
prompts.py   提示词集中管理 + 注入防御  ← 零框架依赖
routing.py   条件边纯函数              ← 零框架依赖
nodes/       业务节点（编排与数据流）
  parse_input.py    需求解析（LLM → InputAnalysis）
  choose_entry.py   interrupt 人机协同（零副作用小节点）
  planning.py       LLM 计划 → 树 的确定性编译（id 分配/链接翻译/幻觉防御）
  modes.py          三入口分支 + fan_out_pages（Send 分发）
  generate_page.py  并行分支本体：单页生成 + 占位回填
  assemble.py       barrier 后收尾验收
graph.py     图组装（框架耦合汇聚点：主图 + 补齐图）
cli.py       终端适配层（Command / get_state / rich 渲染）
preview.py   最小图活文档（单轮演示）
```

依赖方向自上而下；换框架（或升级大版本）时业务层不动，改 graph.py 与 cli.py 两个文件。

### 机制深读

**1. interrupt 不是暂停点，是节点重跑。**
`interrupt()` 抛出 `GraphInterrupt`，图捕获后把上一个已完成节点的状态写入 checkpointer；恢复时（`Command(resume=...)`）该节点**从头重新执行**。推论：interrupt 之前的计算恢复时全部白算 —— 所以 `choose_entry` 是零副作用的独立小节点（读 state → interrupt → 写结果），LLM 解析放在它前一个节点完成并落盘。这也是"interrupt 必须单独拆小节点"的原因：挤在一个节点里，恢复时计费调用翻倍。

**2. Send 是"数据不是引用"。**
`fan_out_pages` 返回 `[Send("generate_page", {"draft": ..., "tree": ...}), ...]`：每个分支收到的是 Send 携带的私有 state（不是全局累积 state），分支之间互不可见。并行安全不靠锁，靠"不共享"；代价是每个分支带一份树快照（数据复制），换来零竞争。归并只有一条路：分支返回只含自己那页的部分树，`merge_tree` 依次折叠。

**3. merge_tree 四律（reducer 是正确性的地基）。**
新 id 直接进树（并行不互相清零）／旧占位 + 新内容 = 回填／旧内容 + 新占位 = 拒绝降级／同级冲突 update 胜。交换律支撑 Send 完成顺序不定；幂等（`merge(t,t)==t`）支撑 checkpointer at-least-once 重放。写错 reducer 不会在定义时报错，而是以"页面丢失/翻倍"的数据损坏形态出现 —— 所以它被 8 个穷举测试锁死。

**4. superstep barrier 是免费的。**
generate_page → assemble 是普通边：N 个并行分支全部完成、归并完毕后，assemble 才在下一个 superstep 执行一次。汇聚是图拓扑的固有语义，不需要手写计数器。`fan_out_pages` 空计划时返回 `["assemble"]`（条件边支持 `list[Send | str]` 混合返回），保证收尾节点总能执行。

**5. 确定性收口：LLM 编意图，代码编机器事实。**
LLM 不输出机器 id（让模型编 `p-001` 是可靠性灾难）—— 规划合同用名字互引，`planning.compile_plan` 做确定性翻译：id 分配（按序编号）、幻觉链接丢弃（计数进 summary，可观测）、页面重名自动改名、未知分区兜底。防御不靠 prompt 恳求，靠编译期校验。

**6. 失败即占位（韧性设计）。**
分解节点先把**全部页面**以 PLACEHOLDER 落进树，生成结果经 reducer 逐页回填。某页 LLM 失败 → 该页留占位、图继续 → 其他页照常生成。占位不是"没做完"，是任何时刻都合法的状态；对它说「补齐」即走补齐图再入同一套生成机制。

### 设计决策速查

| 决策 | 理由 | 放弃的替代方案 |
|------|------|----------------|
| 树用 `dict[id, node]` 扁平存储 | 按 id 归并 O(1)，reducer 与回填的前提 | 嵌套 children（遍历易、归并难） |
| 占位显式建模（NodeStatus） | 状态机 + 类型边界强制不变量 | `content is None` 隐式约定（人人要记得特判） |
| prompt 集中在 prompts.py | 行为资产可 review/diff/回滚；注入防御唯一出口 | 散在节点里（没人敢动） |
| 结构化输出 = Pydantic 校验 + 错误回填重试 | json_object 只兜语法层，语义层必须校验 | 只靠 json_object（枚举拼错照样进来） |
| DI 缝（client=None 参数 + get_client 注入点） | 图级测试走生产路径，零网络零费用进 CI | 全局单例客户端（测试必须 mock 全局） |
| 框架延迟导入、耦合汇聚到 graph.py/cli.py | 业务层可脱离框架穷举测试；换框架只改边缘 | 节点直接 import langgraph（处处是框架） |
| 补齐独立成图复用 generate_page | 同一机制再入，不复制生成逻辑 | 主图加循环（拓扑复杂化，状态纠缠） |

## 依赖清单

运行时：`langgraph>=1.0`（编排）、`langgraph-checkpoint-sqlite>=2.0`（SqliteSaver，2.0 起从主包拆出）、`openai>=1.60`（OpenAI 兼容协议客户端，GLM/Qwen/DeepSeek 均可）、`pydantic>=2.9`、`pydantic-settings>=2.6`、`rich>=13.9`（终端渲染）。
开发：`pytest>=8.3`、`ruff>=0.8`。
在 langgraph 1.2.x 上验证（`InMemorySaver` 为正式名，`MemorySaver` 是兼容别名）。

## 已知限制（v1）

- 补齐结果存于补齐图的独立 thread，不回写主图 thread；CLI 重启后 `get_state(主thread)` 拿到的是主图最后一棵树。树会导出到 `.shop_agent/trees/*.json` 供检查。
- Send 并行为逻辑并行（sync 图内按 superstep 调度），物理并发受限于运行器；页面级并行已足够本场景。
- 手动模式仅提供会话待命，指令式建树（"给我加一个活动页"）未实现 —— 状态层的 merge_tree 已支持该扩展（归并是加法不是替换）。

## 声明

本项目为个人学习作品。灵感来自作者实习期间接触的多 Agent 编排模式；因保密协议，本仓库全部代码基于公开框架（LangGraph）与公开文档独立实现，不含任何公司代码、文档与业务数据。

## License

MIT
