# 科研智能助手 Agent

## 01 技术栈

Python、FastAPI、LangGraph、ChromaDB、Redis、SQLite、WebSocket、RAG、BM25、RRF

## 02 项目简介

面向论文调研与科研知识问答场景，独立设计并实现科研智能助手 Agent。系统能够理解用户研究目标，自主拆解任务并调用知识库、论文检索、联网搜索等工具，完成从资料召回、分析整理到答案生成的完整流程；同时支持多知识库隔离、长短期记忆、多 Agent 协作、失败恢复与端到端质量评测。

## 03 项目亮点（简历推荐版）

* **智能任务执行**：实现 ReAct 单 Agent 执行引擎，并基于 LangGraph 构建 Supervisor-Worker 多 Agent 工作流，由 Researcher、Analyst、Writer 分工完成资料搜集、分析与成稿，通过共享黑板传递阶段成果，减少重复处理。

* **混合知识检索**：支持用户创建和维护多个独立知识库，对论文等资料进行解析、切分和向量化；采用向量检索与 BM25 双路召回、RRF 融合及可插拔二阶段重排，提高专业术语和语义查询的检索稳定性。

* **可靠工具调用**：建立确定性路由与模型自主决策结合的工具选择策略，支持知识库、arXiv、联网搜索、PDF 阅读和计算等工具；针对超时、参数错误、重复调用和空结果设计有限重试、熔断及自动降级机制。

* **复杂任务容错**：针对 Agent 死循环设计调用去重、结果指纹、跨步骤 LoopGuard 与 RecoveryPolicy；检测无进展后自动切换“知识库 → arXiv → 联网搜索”，所有路径失败时保存任务暂停点，请求用户补充文档或检索条件，并通过一次性令牌从原步骤继续执行。

* **记忆与质量保障**：使用 Redis + SQLite 管理会话热数据和持久化备份，结合 ChromaDB 实现跨会话事实召回；建立任务完成率、回答正确率、事实依据度、工具调用成功率和延迟等评测，并完成 **315 项自动化测试**。

## 04 一句话面试介绍

这是一个面向科研场景的 RAG Agent：它不仅能从个人知识库和论文网站查资料，还能拆解复杂任务、协调多个专家 Agent，并在工具超时、检索无结果或执行陷入循环时自动换路；如果仍缺少必要资料，会保存当前进度并请求用户补充，之后从原步骤继续。

## 05 适合解决方案岗位的项目价值

* 将“上传资料—知识检索—任务执行—结果交付”串成完整业务闭环，而不是单一聊天功能。
* 通过用户级、知识库级数据隔离支持不同研究人员独立使用。
* 通过可观测事件流展示任务计划、工具调用、执行结果与异常恢复过程，便于解释系统如何得出结论。
* 通过自动降级和断点续跑提高外部模型、向量数据库或搜索服务异常时的可用性。

---

# 面试知识整理

下面内容用于面试复习，不建议全部放入正式简历。


RRF 融合排序，你可以简单理解成：

**把“向量检索”和“BM25 检索”各自得到的排名结果合并，重新排出一个最终顺序。**

这个项目的 RRF 用于把两路检索结果合并：

1.  向量检索：寻找“语义相近”的内容。
    
2.  BM25：寻找“关键词匹配”的内容。
    
3.  RRF：不直接比较两种检索分数，而是根据各自的排名计算综合得分。
    
4.  可选重排：再对融合结果精排，最终返回 Top K。
    

核心公式是：

```plaintext
RRF 分数 = Σ 1 / (k + rank)   RRF 分数 = Σ 1 / (k + 排名)
```

项目里 `k=60`，代码排名从 `0` 开始，所以实际实现为：

```plaintext
1 / (60 + rank + 1)
```

例如某篇论文：

*   向量检索排第 1：`1 / 61`
    
*   BM25 检索排第 2：`1 / 62`
    
*   综合得分：`1/61 + 1/62 ≈ 0.0325`
    

如果另一篇论文只在向量检索中排第 1，它的得分只有 `1/61 ≈ 0.0164`。因此，两路检索都认可的内容通常会排得更靠前。

项目中的具体流程是：

```plaintext
用户问题
   ↓
查询改写
   ↓
向量检索 + BM25 检索
   ↓
分别按 doc_id 去重
   ↓
RRF 按排名融合
   ↓
重排器精排
   ↓
截取 Top K，交给大模型生成答案
```

核心实现位于 \[fusion.py (line 10)\](D:\\科研助手agent\\app\\rag\\fusion.py:10)，实际调用位于 \[pipeline.py (line 162)\](D:\\科研助手agent\\app\\rag\\pipeline.py:162)。

它的主要价值是：向量分数和 BM25 分数的计算方式不同，不能直接相加；RRF 只看排名，因此无需人工归一化分数，对专业术语、缩写和语义表达都有更稳定的召回效果。

面试时可以简洁地说：

> 项目采用向量检索与 BM25 双路召回，再通过 RRF 基于排名进行融合，避免不同检索分数无法直接比较的问题，使同时被语义和关键词检索命中的资料获得更高优先级，最后通过重排模型进一步筛选结果。

可以把它拆成两层理解：**ReAct 负责“一个 Agent 怎么做事”，LangGraph 负责“多个 Agent 怎么协作”**。

## 1. ReAct 是如何设计的

ReAct 的含义是 Reasoning + Acting，也就是循环执行：

```plaintext
理解问题 → 决定行动 → 调用工具 → 读取结果 → 继续判断 → 输出答案
```

项目中的具体流程是：

1.  用户提出问题。
    
2.  Agent 读取系统提示、会话历史和长期记忆。
    
3.  大模型判断是否需要调用工具。
    
4.  如果需要，就生成 `tool_calls`。
    
5.  系统并行执行知识库检索、论文搜索、计算器等工具。
    
6.  工具结果作为 Observation 写回消息历史。
    
7.  大模型根据新信息继续推理。
    
8.  不再调用工具时，输出最终答案。
    

例如用户问：“查找 Transformer 在医学影像领域的研究并总结”：

```plaintext
Thought：需要先查找相关论文
Action：调用论文搜索
Observation：返回论文列表
Thought：需要补充用户知识库中的相关材料
Action：调用知识库检索
Observation：返回相关文档片段
Thought：信息已经足够
Answer：生成带依据的总结
```

项目还给这个循环增加了工程保护：

*   最大循环次数，避免 Agent 无限执行。
    
*   最大工具调用次数，控制成本。
    
*   工具超时与失败重试。
    
*   上下文长度控制和历史消息压缩。
    
*   工具结果过长时自动截断。
    
*   达到限制后强制根据已有信息生成答案。
    
*   支持流式展示思考、工具调用和执行结果。
    

主要实现在 \[agent.py (line 126)\](D:\\科研助手agent\\app\\agent\\agent.py:126)。

## 2. LangGraph 是如何设计的

项目中 LangGraph 有两种用途。

### 单 Agent 图

它把 ReAct 循环显式表示成两个节点：

```plaintext
START
  ↓
reason（大模型判断）
  ├─ 有工具调用 → tools（执行工具）→ reason
  └─ 无工具调用 → END
```

它和手写 ReAct 的逻辑基本等价，只是把循环改成了状态图，实现在 \[single\_agent.py (line 48)\](D:\\科研助手agent\\app\\graph\\single\_agent.py:48)。

### 多 Agent 协作图

项目采用 Supervisor + 专家 Agent 的模式：

```plaintext
                    ┌→ Researcher ─┐
用户问题 → Supervisor → Analyst ───→ Supervisor → 最终答案
                    └→ Writer ─────┘
```

三个专家的职责分别是：

*   `Researcher`：调用知识库、论文搜索、PDF 阅读等工具收集事实。
    
*   `Analyst`：负责计算分析、数据处理和引用规范化。
    
*   `Writer`：整合已有结果，生成结构化最终答案。
    
*   `Supervisor`：判断下一步应该把任务交给哪个专家，或者直接结束任务。
    

工具权限也做了隔离：

| Agent | 可以使用的工具 |
| --- | --- |
| Researcher | 知识库、论文检索、PDF 阅读、代码执行 |
| Analyst | 计算器、引用工具、代码执行 |
| Writer | 不调用工具，只负责整理输出 |

多 Agent 主流程是：

```plaintext
Supervisor 判断任务
        ↓
通过 handoff 移交给专家
        ↓
专家执行工具并产出中间结果
        ↓
结果写入共享黑板
        ↓
返回 Supervisor
        ↓
继续委派或生成最终答案
```

主要实现在 \[multi\_agent.py (line 108)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:108)。

## 3. Agent 之间如何传递信息

项目设计了三种状态：

*   `messages`：完整的对话和工具调用记录。
    
*   `blackboard`：共享黑板，保存每位专家已经完成的核心结论。
    
*   `inbox`：定向传递给下一位专家的任务和上游产物。
    

例如：

```plaintext
Researcher 找到论文
    ↓
结果写入共享黑板
    ↓
Supervisor 将结果交给 Analyst
    ↓
Analyst 完成数据分析
    ↓
Supervisor 将研究和分析结果交给 Writer
    ↓
Writer 生成最终报告
```

这种设计可以减少重复搜索，也能让前端展示任务由谁处理、调用了什么工具以及当前进展。

需要注意的是，简历原句容易让人理解成“ReAct 和 LangGraph 必须同时运行”。更准确的表达是：

> 设计自研 ReAct 单 Agent 执行引擎，并基于 LangGraph 实现 Supervisor 多 Agent 编排，支持知识检索、论文搜索等工具的自主调用，以及复杂任务的角色分工与多步骤执行。

这个项目的“共享黑板”本质上是 **LangGraph 状态中的一个字典**，用来保存各个专家 Agent 的阶段性成果。

## 1. 数据结构

核心状态定义如下：

```plaintext
class TeamState(MessagesState):
    blackboard: dict = {}
    inbox: dict = {}
    chain: list = []
```

三部分职责不同：

| 状态 | 作用 |
| --- | --- |
| `blackboard` | 保存每个专家的最新结论 |
| `inbox` | 把上游成果定向交给下一个专家 |
| `chain` | 记录 Agent 之间的交接顺序 |

例如执行过程中，状态可能是：

```plaintext
{
  "blackboard": {
    "researcher": "检索到 8 篇相关论文，主要分为三个研究方向……",
    "analyst": "根据论文数据，对比结果显示方法 A 的准确率更高……",
    "writer": "最终形成的结构化研究综述……"
  },
  "chain": [
    {"from": null, "to": "researcher"},
    {"from": "researcher", "to": "analyst"},
    {"from": "analyst", "to": "writer"}
  ]
}
```

代码位于 \[multi\_agent.py (line 95)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:95)。

## 2. 专家如何写入黑板

每个 Worker 完成任务后，把最终产出写入以自身角色命名的位置：

```plaintext
return {
    "messages": out,
    "blackboard": {
        "researcher": "本轮研究结果……"
    }
}
```

不同 Agent 分别写入：

```plaintext
Researcher → blackboard["researcher"]
Analyst   → blackboard["analyst"]
Writer    → blackboard["writer"]
```

黑板使用合并函数更新：

```plaintext
def _merge_blackboard(old, new):
    return {**old, **new}
```

因此：

*   不同专家的结果会保留下来。
    
*   同一个专家再次执行时，新结果覆盖旧结果。
    
*   不会因为某个节点只返回局部数据而清空整个黑板。
    

写入逻辑位于 \[multi\_agent.py (line 215)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:215)。

## 3. Supervisor 如何读取黑板

每次专家完成任务后，流程都会返回 Supervisor。

Supervisor 会把黑板内容拼入系统提示：

```plaintext
【共享黑板】各专家已写入的进展：
- researcher: 已找到相关论文……
- analyst: 已完成实验数据对比……

请基于黑板进展决定下一步委派。
```

然后 Supervisor 决定：

*   信息不足：继续交给 Researcher。
    
*   需要处理数据：交给 Analyst。
    
*   信息已经充分：交给 Writer。
    
*   已经得到完整答案：结束流程。
    

这避免了 Supervisor 只根据原始问题盲目分配任务。读取逻辑位于 \[multi\_agent.py (line 162)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:162)。

## 4. 为什么还需要 Inbox

黑板是所有 Agent 的公共区域，但项目还设计了 `inbox` 做定向交接。

例如：

```plaintext
Researcher 完成论文检索
        ↓
写入共享黑板
        ↓
Supervisor 选择 Analyst
        ↓
将 Researcher 的结论放进 Analyst 的 inbox
        ↓
Analyst 直接基于检索结果分析
```

Worker 收到任务时，系统会将 Inbox 注入提示词：

```plaintext
【上一专家 researcher 直接交付给你的核心结论】
已检索到 8 篇论文，关键数据如下……

请直接使用，不要重复检索或计算。
```

因此两者的区别是：

*   `blackboard`：团队共享的成果存储区。
    
*   `inbox`：给特定 Agent 的任务交接包。
    

定向传递位于 \[multi\_agent.py (line 180)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:180)。

## 5. 是否支持跨会话保存

单次 LangGraph 运行中，黑板保存在 `TeamState` 里。

任务结束后，如果传入了 `session_id`，项目会把黑板内容作为事实写入长期记忆：

```plaintext
{
    "category": "blackboard",
    "content": "researcher: 检索结论……",
    "tags": ["researcher"]
}
```

所以设计上分成两层：

```plaintext
运行中的共享黑板
        ↓ 任务结束
长期记忆持久化
        ↓
后续会话可以再次召回
```

持久化代码位于 \[multi\_agent.py (line 292)\](D:\\科研助手agent\\app\\graph\\multi\_agent.py:292)。

## 当前实现的一个问题

Supervisor 的 handoff 支持携带 `payload`，但 Worker 构建提示词时把 `_payload_` 跳过了，导致主管的定向指令实际上没有注入 Worker 提示词：

```plaintext
if k in ("_payload_", "_prev_member_", "_prev_product_"):
    continue
```

后面的这句也因此不会执行：

```plaintext
label = "主管指令" if k == "_payload_" else ...
```

所以目前“专家成果直传”可以正常工作，但“主管附加指令”存在实现缺口，建议修复。

面试时可以概括为：

> 基于 LangGraph State 设计共享黑板，将不同专家的阶段性成果按角色合并存储；Supervisor 根据黑板动态决策下一步任务，并通过 Inbox 将上游成果定向传递给下游 Agent，减少重复检索，任务结束后再将关键结论沉淀到长期记忆。

项目采用的是分层记忆设计，可以理解为：

```plaintext
L1 短期记忆：记住当前会话聊了什么
L2 长期记忆：记住跨会话仍有价值的事实
L3 会话索引：管理历史会话及摘要
```

## 1. 短期记忆

短期记忆保存当前会话中的完整消息，包括：

*   用户问题
    
*   Agent 回答
    
*   工具调用参数
    
*   工具执行结果
    
*   会话标题和摘要
    

执行流程是：

```plaintext
用户发送消息
    ↓
写入当前 session_id
    ↓
Agent 读取最近的会话历史
    ↓
组合进 Prompt
    ↓
大模型理解上下文并继续回答
```

为了防止历史记录无限增长，项目实现了：

*   默认保留最近 20 条消息。
    
*   根据 Token 预算动态截断。
    
*   长对话自动压缩成摘要。
    
*   保证 `tool_call` 和对应的工具结果不会被拆开。
    

核心实现在 \[conversation.py (line 15)\](D:\\科研助手agent\\app\\memory\\conversation.py:15)。

### 短期记忆存储

项目支持三种后端：

*   内存：开发使用，服务重启后丢失。
    
*   SQLite：本地持久化，重启后仍然保留。
    
*   Redis + SQLite：Redis 保存热数据，SQLite 做持久化兜底。
    

当前 `.env` 最终生效的配置是：

```plaintext
MEMORY__BACKEND=redis
MEMORY__REDIS_URL=redis://127.0.0.1:6379/0
```

所以当前设计是：

```plaintext
写入消息
 ├─ Redis：快速读取、支持 TTL
 └─ SQLite：同步持久化、防止 Redis 数据丢失

读取消息
 ├─ 优先读取 Redis
 └─ Redis 未命中 → SQLite 恢复 → 回填 Redis
```

这比单独使用 Redis 更可靠。实现位于 \[persistence.py (line 23)\](D:\\科研助手agent\\app\\memory\\stores\\persistence.py:23)。

## 2. 长期记忆

长期记忆不保存全部对话，而是由大模型在回答结束后，提取最多三条值得长期保留的信息，例如：

```plaintext
[
  {
    "content": "用户研究方向是医学图像分割",
    "category": "user_fact"
  },
  {
    "content": "本次研究发现 U-Net 更适合小样本场景",
    "category": "finding"
  }
]
```

主要提取：

*   用户研究方向
    
*   用户偏好
    
*   已确认的研究结论
    
*   使用过的方法或工具
    
*   对后续任务有价值的信息
    

提取逻辑位于 \[finalizer.py (line 101)\](D:\\科研助手agent\\app\\agent\\finalizer.py:101)。

### 长期记忆存储

长期事实的主存储是 **ChromaDB**：

```plaintext
事实文本 → Embedding 向量化
        → 写入 ChromaDB 的 memory_facts 集合
```

保存的数据包括：

```plaintext
document：事实原文
metadata：
  - session_id
  - category
  - tags
  - created_at
vector：事实的语义向量
```

当 ChromaDB 不可用时，会自动降级到：

```plaintext
SQLite facts 表
    ↓
优先使用 FTS5 全文检索
    ↓
FTS5 不可用或中文未命中时使用 LIKE
```

实现位于 \[longterm.py (line 34)\](D:\\科研助手agent\\app\\memory\\longterm.py:34)。

## 3. 长期记忆如何召回

用户提出新问题后，系统使用最新问题检索长期事实：

```plaintext
当前问题
   ↓
生成查询向量
   ↓
在 ChromaDB 中语义检索
   ↓
取最相关的 3 条事实
   ↓
注入系统提示词
```

注入后的内容类似：

```plaintext
## 跨会话记忆
- 用户的研究方向是医学图像分割
- 用户更关注小样本训练方法
- 上次分析认为 U-Net 更适合当前数据集

如与当前问题冲突，以当前对话为准。
```

每轮任务只注入一次，避免在 ReAct 循环中反复追加相同内容。召回逻辑位于 \[context.py (line 65)\](D:\\科研助手agent\\app\\agent\\context.py:65)。

## 4. 会话索引

项目还使用 SQLite 保存：

*   会话 ID
    
*   会话标题
    
*   会话摘要
    
*   创建时间
    
*   更新时间
    

它主要用于展示历史会话、搜索和恢复记录，不属于语义记忆本身。

## 面试表达

> 项目采用分层记忆架构：短期记忆通过 Redis 缓存当前会话消息，并双写 SQLite 实现持久化和故障恢复；长期记忆由大模型从对话中提取用户偏好与研究结论，向量化后存入 ChromaDB，通过语义检索实现跨会话召回，ChromaDB 不可用时自动降级至 SQLite 全文检索。同时通过滑动窗口、Token 截断和会话摘要控制上下文长度。

Redis 不一定会丢数据，但它的主要定位是**高性能内存数据库**。相比 SQLite 这类磁盘数据库，在下面几种情况下更容易出现数据丢失。

## 1. Redis 服务重启

数据主要保存在内存中。如果没有开启持久化，Redis 重启后内存数据会被清空：

```plaintext
Redis 运行中：会话数据存在
        ↓
服务重启或电脑关机
        ↓
内存释放
        ↓
会话数据丢失
```

## 2. Key 到期

项目给 Redis 会话设置了 TTL。超过保存时间后，Redis 会自动删除对应 Key：

```plaintext
agent:memory:session_001
        ↓ TTL 到期
自动删除
```

这不属于故障，而是缓存的正常清理机制。

## 3. Redis 内存不足

当 Redis 达到内存上限时，根据 `maxmemory-policy` 配置，可能会淘汰部分 Key，例如：

*   优先删除即将过期的数据。
    
*   删除最久未使用的数据。
    
*   随机删除数据。
    

如果会话数据被选中，就会从 Redis 消失。

## 4. Redis 持久化存在时间窗口

即使开启了持久化，也不代表绝对不会丢：

*   RDB：定期生成快照，故障时可能丢失上次快照之后的数据。
    
*   AOF：记录写命令；如果每秒同步一次，极端情况下可能丢失约一秒的数据。
    
*   未开启 RDB 和 AOF：重启后数据可能全部消失。
    

## 项目如何解决

项目采用 Redis + SQLite 双写：

```plaintext
新会话消息
   ├─ 写入 Redis：读取速度快
   └─ 写入 SQLite：磁盘持久化
```

读取时：

```plaintext
先查 Redis
   ├─ 命中 → 直接返回
   └─ 未命中 → 从 SQLite 读取 → 回填 Redis
```

所以这里不是认定“Redis 一定会丢数据”，而是把 Redis 当成高速热数据层，把 SQLite 当成可靠持久化层。

面试时可以说：

> Redis 负责会话热数据的快速访问，但考虑到 TTL 过期、内存淘汰和服务重启等情况，系统将消息同步写入 SQLite；当 Redis 未命中时，会从 SQLite 恢复并自动回填 Redis，在性能和可靠性之间取得平衡。

RRF 当然可以单独使用。加入 Reranker，是为了弥补 RRF **只看排名、不看内容是否真正回答问题**的缺点。

假设用户问：

> 苹果公司的芯片有什么优势？

两路检索结果如下：

| 文档 | 向量排名 | BM25 排名 |
| --- | --- | --- |
| A：苹果公司的水果供应链 | 2 | 1 |
| B：Apple M 系列芯片性能分析 | 4 | 5 |

RRF 只看到排名：

*   A 在两路都靠前，因此 RRF 分数高。
    
*   B 排名稍后，因此 RRF 分数低。
    

于是 RRF 可能把 A 排在前面。但 A 中的“苹果”指水果，并没有真正回答芯片问题。

Reranker 会把查询和文档内容放在一起判断：

```plaintext
查询：苹果公司的芯片有什么优势？
文档A：水果产地、运输和销售……
→ 相关性低

查询：苹果公司的芯片有什么优势？
文档B：Apple M 系列采用统一内存架构……
→ 相关性高
```

最终 Reranker 会把 B 调到 A 前面。

核心区别是：

```plaintext
RRF：
“这篇文档在多个检索器里排第几？”

Reranker：
“这篇文档的内容究竟能不能回答这个问题？”
```

为什么不直接只用 Reranker？因为让它比较知识库中的全部文档，速度慢、成本高。所以通常采用两阶段结构：

```plaintext
第一阶段：快速召回
向量检索 + BM25
例如从 10 万篇文档中找出 50 篇
        ↓
RRF 融合
把两路结果合并、去重、排序
        ↓
第二阶段：精准筛选
Reranker 阅读这 50 篇候选
选出最相关的 5 篇
```

可以把它类比成招聘：

*   向量检索和 BM25：两个招聘网站分别推荐候选人。
    
*   RRF：根据候选人在两个网站推荐榜中的名次，合并简历。
    
*   Reranker：HR 真正阅读候选人的经历，判断是否匹配岗位。
    

因此：

*   对简单、明确的查询，RRF 通常已经够用。
    
*   对歧义词、复杂问题、专业文献，Reranker 更有价值。
    
*   Reranker 不是必须的，是一个用计算成本换准确率的精排环节。
    

一句话概括：

> RRF 根据“排名共识”筛候选，Reranker 根据“问题与内容的真实相关性”纠正排名。

项目里实现了两种 Reranker，但**当前默认使用的是轻量词面重排器，不是大模型重排器**。

当前配置：

```plaintext
reranker_kind = "lexical"
rerank_vector_weight = 0.7
rerank_lexical_weight = 0.3
```

## 当前默认实现

RRF 合并出候选文档后，Reranker 对每篇文档重新计算：

```plaintext
重排分数 =
0.7 × 归一化向量相似度
+
0.3 × 查询词与文档词的重合数量
```

对应代码：

```plaintext
query_tokens = tokenize(query)

vector_score = candidate["score"] / max_vector_score
overlap = len(query_tokens & document_tokens)

final_score = 0.7 * vector_score + 0.3 * overlap
```

然后根据 `final_score` 从高到低重新排序：

```plaintext
return sorted(candidates, key=score, reverse=True)
```

例如用户问：

```plaintext
Transformer 在医学影像分割中的应用
```

RRF 返回三个候选：

| 文档 | 向量相似度 | 关键词重合 |
| --- | --- | --- |
| A：Transformer 通用介绍 | 0.90 | 1 |
| B：Transformer 医学影像分割 | 0.82 | 4 |
| C：医学图像传统分割方法 | 0.75 | 2 |

重新计算后，B 虽然原始向量排名不是第一，但因为与问题中的关键词更匹配，可能被调整到第一。

流程就是：

```plaintext
RRF 候选结果
    ↓
提取问题关键词
    ↓
计算每篇文档的向量分 + 关键词重合分
    ↓
按加权分数重新排序
    ↓
截取最终 Top K
```

实现位于 \[rerank.py (line 44)\](D:\\科研助手agent\\app\\knowledge\\rerank.py:44)，调用位于 \[pipeline.py (line 180)\](D:\\科研助手agent\\app\\rag\\pipeline.py:180)。

## 可选的 Cross-Encoder 实现

项目还支持更标准的语义 Reranker：

```plaintext
CrossEncoder("BAAI/bge-reranker-v2-m3")
```

它会把每个问题和候选文档组成一对：

```plaintext
[
    (query, document_A),
    (query, document_B),
    (query, document_C)
]
```

然后模型直接输出相关性分数：

```plaintext
问题 + 文档 A → 0.42
问题 + 文档 B → 0.91
问题 + 文档 C → 0.28
```

最后按照分数重新排序。

它和向量检索最大的区别是：

*   向量检索：问题和文档分别编码，再计算向量距离，速度快。
    
*   Cross-Encoder：问题和文档一起输入模型，让模型逐字判断相关性，准确但更慢。
    

把配置改为下面这样即可启用：

```plaintext
KNOWLEDGE__RERANKER_KIND=cross-encoder
```

不过需要安装 `sentence-transformers` 并下载模型。

## 当前实现需要注意的问题

你现在默认的 `LexicalReranker` 更准确地说是**轻量规则重排**，不能描述成“基于深度模型的语义精排”。

而且它目前存在两个可优化点：

1.  计算时使用的是候选文档原始向量分数，没有真正把 `rrf_score` 加入重排公式。
    
2.  中文分词比较简单，连续中文可能被当成一个整体词，关键词重合效果有限。
    

因此，简历上目前最准确的说法是：

> 在向量与 BM25 双路召回后使用 RRF 融合候选结果，并结合向量相关度和关键词匹配进行二次重排；同时预留 Cross-Encoder 语义精排能力。

不要直接写“使用 Cross-Encoder 重排”，因为项目虽然实现了该功能，但当前默认配置并未启用。

对，当前默认重排逻辑中，**语义相似度的优先级更高**。

计算公式是：

```plaintext
最终得分 = 0.7 × 向量相似度 + 0.3 × 关键词重合度
```

因此：

*   向量相似度占 `70%`：主要判断语义是否相关。
    
*   关键词匹配占 `30%`：用于补充和纠偏。
    
*   最后根据新的综合得分重新排序。
    

完整流程可以理解为：

```plaintext
向量检索 + BM25
        ↓
RRF 合并两路候选
        ↓
重排阶段重新强调语义相似度
        ↓
关键词匹配辅助纠偏
        ↓
返回最终 Top K
```

不过有个关键细节：当前代码的 Reranker 使用的是**原始向量分数**，没有使用 `rrf_score`。所以 RRF 主要负责合并和扩大候选集，最终顺序更受“70% 向量相似度 + 30% 关键词匹配”影响。

一句话总结：

> 是的，当前重排器会重新提高语义相似度的权重，同时保留一部分关键词匹配能力。

对，**当前项目的 RRF 可以理解为向量检索和 BM25 各占 50%**，因为两路使用完全相同的计算公式，没有额外权重

可以把这句话拆成以下几个概念：

## 1. RRF 融合

RRF 全称 `Reciprocal Rank Fusion`，中文叫“倒数排名融合”。

它用来合并向量检索和 BM25 返回的两份排行榜。它不直接比较两路原始分数，只比较排名。

计算公式：

```plaintext
RRF(doc) = Σ 1 / (k + rank)
```

项目中 `k=60`，代码排名从 0 开始，因此实际计算为：

```plaintext
RRF(doc) = Σ 1 / (60 + rank + 1)
```

假设文档 A：

*   向量检索排第 1
    
*   BM25 排第 3
    

那么：

```plaintext
RRF(A) = 1/61 + 1/63
       ≈ 0.0323
```

文档 B：

*   向量检索排第 2
    
*   BM25 没有召回
    

那么：

```plaintext
RRF(B) = 1/62
       ≈ 0.0161
```

所以文档 A 排在 B 前面。

要点：

*   排名数字越小越好。
    
*   RRF 最终分数越高越好。
    
*   当前两路检索等权。
    
*   两路都靠前的文档会得到更高分数。
    

---

## 2. 候选结果

“候选结果”指第一阶段从知识库里快速找出来的一批可能相关的文档。

例如知识库有 10 万个文档：

```plaintext
向量检索召回 15 条
BM25 召回 15 条
        ↓
合并并按文档 ID 去重
        ↓
得到约 20～30 条候选文档
```

它们只是“可能相关”，还不是最终交给大模型的资料。

项目候选数量大致按照下面的方式确定：

```plaintext
candidate_k = max(最终数量 × 3, 重排候选数, 最终数量)
```

例如最终需要返回 5 条：

```plaintext
candidate_k = max(5 × 3, 3, 5) = 15
```

向量检索和 BM25 分别最多召回 15 条，然后交给 RRF 融合。

---

## 3. 二阶段重排

“二阶段”指检索被分成两个阶段。

### 第一阶段：快速召回

```plaintext
向量检索 + BM25 → RRF 融合
```

目标是从大量文档中快速找出一批候选，重点是尽量不要漏掉相关资料。

### 第二阶段：精细排序

```plaintext
RRF 候选 → Reranker → 最终 Top K
```

目标是判断候选文档中，哪些和问题更加相关，并重新排列顺序。

当前默认轻量重排公式是：

```plaintext
重排分数 =
0.7 × 归一化向量相似度
+
0.3 × 关键词重合数量
```

其中：

```plaintext
归一化向量分数 =
当前文档向量分数 / 候选中的最高向量分数
```

例如：

```plaintext
文档向量分数：0.72
最高向量分数：0.80

归一化结果 = 0.72 / 0.80 = 0.9
```

如果关键词重合分经过简化后为 `0.6`：

```plaintext
最终分数 = 0.7 × 0.9 + 0.3 × 0.6
         = 0.81
```

然后按最终分数从高到低排序，取前 5 条交给大模型。

---

## 4. Recall@K

Recall@K 叫“前 K 条召回率”，衡量：

> 所有应该找到的正确文档中，系统在前 K 条里找到了多少？

公式：

```plaintext
Recall@K =
前 K 条中正确文档数量 / 全部正确文档数量
```

假设标准答案中有 4 篇相关论文，系统前 5 条找到了其中 3 篇：

```plaintext
Recall@5 = 3 / 4 = 0.75
```

含义是：系统找回了 75% 的相关资料。

*   越接近 `1` 越好。
    
*   主要衡量“找得全不全”。
    
*   不太关心正确结果具体排第几。
    

---

## 5. MRR@K

MRR 全称 `Mean Reciprocal Rank`，衡量：

> 第一篇正确文档出现得够不够靠前？

单个问题的计算公式：

```plaintext
RR = 1 / 第一篇正确文档的排名
```

例如：

| 第一篇正确文档位置 | RR |
| --- | --- |
| 第1名 | 1 |
| 第2名 | 0.5 |
| 第3名 | 0.333 |
| 第5名 | 0.2 |
| 前K名没有 | 0 |

多个问题取平均：

```plaintext
MRR@K = 所有问题 RR 之和 / 问题数量
```

假设三个问题的第一篇正确文档分别排在第 1、2、4 名：

```plaintext
MRR@5 = (1 + 1/2 + 1/4) / 3
      = 0.583
```

*   越接近 `1` 越好。
    
*   主要看第一个正确结果的位置。
    
*   不关心后面还有多少正确结果。
    

---

## 6. nDCG@K

nDCG 全称 `Normalized Discounted Cumulative Gain`，衡量：

> 整体排序是否合理，高相关资料是否排在前面？

它允许相关性分级，例如：

```plaintext
3分：高度相关
2分：比较相关
1分：弱相关
0分：不相关
```

首先计算 DCG：

```plaintext
DCG@K = Σ (2^相关性分数 - 1) / log₂(排名 + 1)
```

之所以除以对数，是因为越靠后的文档贡献越小。

例如前三条文档的相关性分别是：

```plaintext
[3, 0, 2]
```

则：

```plaintext
DCG@3 =
(2³-1)/log₂(2)
+
(2⁰-1)/log₂(3)
+
(2²-1)/log₂(4)

= 7 + 0 + 1.5
= 8.5
```

然后计算理想排序 `IDCG`。理想顺序应该是：

```plaintext
[3, 2, 0]
```

最后归一化：

```plaintext
nDCG@K = DCG@K / IDCG@K
```

取值通常为 `0～1`：

*   `1`：排序与理想顺序一致。
    
*   越接近 `0`：排序越差。
    

与 MRR 的区别：

*   MRR 只看第一条正确结果。
    
*   nDCG 观察前 K 条的整体顺序和相关程度。
    

---

## 7. 平均检索延迟

平均检索延迟衡量：

> 从提交查询到检索链路返回文档，平均需要多长时间？

通常计时范围包括：

```plaintext
查询向量化
+ 向量检索
+ BM25 检索
+ RRF 融合
+ Reranker 重排
```

不包含大模型最终生成答案的时间。

计算公式：

```plaintext
平均检索延迟 =
所有查询的检索耗时总和 / 查询数量
```

例如：

```plaintext
查询1：60ms
查询2：80ms
查询3：70ms

平均延迟 = (60 + 80 + 70) / 3
         = 70ms
```

延迟越低越好，但必须结合质量指标判断：

```plaintext
方案 A：nDCG=0.90，延迟=70ms
方案 B：nDCG=0.91，延迟=18000ms
```

方案 B 虽然质量略高，但实时使用体验很差，通常不会作为默认方案。

---

## 8. 消融评测

消融评测不是一个计算公式，而是一种实验方法：

> 每次只改变或关闭一个模块，观察质量和速度发生什么变化。

例如对比：

| 实验方案 | 目的 |
| --- | --- |
| 仅向量检索 | 建立基础结果 |
| 向量 + BM25 | 验证 BM25 是否有价值 |
| 向量 + BM25 + RRF | 验证融合是否有效 |
| RRF + Lexical 重排 | 验证轻量重排效果 |
| RRF + Cross-Encoder | 验证语义精排效果 |

每个方案使用同一份语料、同一组问题和同一套标准答案，比较：

```plaintext
Recall@K
MRR@K
nDCG@K
平均检索延迟
```

这样才能证明某个模块是否真的有用，而不是仅凭感觉判断。

---

## 9. Cross-Encoder

Cross-Encoder 是一种语义重排模型。

它把“问题”和“候选文档”同时输入模型：

```plaintext
[问题, 文档] → Cross-Encoder → 相关性分数
```

例如：

```plaintext
问题：Transformer 如何用于医学影像分割？
文档A：介绍 Transformer 的自然语言处理应用
→ 0.42

文档B：介绍基于 Transformer 的医学影像分割模型
→ 0.93
```

然后按照相关性分数从高到低重新排序：

```plaintext
rerank_score 越高，文档越靠前
```

它比普通向量检索更精确，是因为：

*   向量检索分别编码问题和文档。
    
*   Cross-Encoder 同时阅读问题和文档，可以分析二者之间更细致的关系。
    

缺点是每篇候选文档都要单独计算，速度明显更慢。

---

## 10. 可插拔精排

“可插拔”表示 Reranker 具有统一接口，可以通过配置切换，不需要修改检索主流程。

项目支持：

```plaintext
lexical       → 轻量规则重排
cross-encoder → 深度语义精排
none          → 不进行重排
```

例如：

```plaintext
KNOWLEDGE__RERANKER_KIND=lexical
```

切换为：

```plaintext
KNOWLEDGE__RERANKER_KIND=cross-encoder
```

检索主流程仍然是：

```plaintext
召回 → RRF → reranker.rerank() → Top K
```

只替换具体的重排器实现。

---

## 11. 失败自动降级

Cross-Encoder 可能因为以下原因无法运行：

*   模型没有下载成功。
    
*   网络不可用。
    
*   缺少 `sentence-transformers` 或 PyTorch。
    
*   内存不足。
    
*   模型推理异常。
    

项目会捕获这些异常，然后切换到轻量重排器：

```plaintext
尝试 Cross-Encoder
        ↓
是否成功？
  ├─ 是 → 使用语义精排结果
  └─ 否 → 自动改用 LexicalReranker
```

降级之后，本次 RAG 检索仍能返回结果，不会因为精排模型失败而让整个问答请求报错。

它没有复杂公式，本质是一个容错策略：

```plaintext
try:
    return cross_encoder.rerank(query, candidates)
except Exception:
    return lexical_reranker.rerank(query, candidates)
```

一句话总结整条链路：

> 向量检索和 BM25 负责尽可能找全资料，RRF 根据两路排名合并候选，Reranker 对候选做第二轮排序；再通过 Recall 衡量找得全不全、MRR 衡量第一条正确结果是否靠前、nDCG 衡量整体排序质量，并结合延迟选择最合适的生产配置。
