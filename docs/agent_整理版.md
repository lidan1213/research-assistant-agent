# 科研智能助手 Agent：项目说明与面试手册

> 本文按照项目真实运行顺序整理，可用于复习、项目讲解和面试准备。内容以当前代码为准。

## 1. 简历版信息

### 1.1 技术栈

Python、FastAPI、LangChain、LangGraph、RAG、FAISS、ChromaDB、Redis、SQLite、WebSocket/SSE、pytest

### 1.2 项目简介

科研智能助手 Agent 面向论文调研和科研知识问答场景，支持用户创建独立知识库、上传论文并进行基于资料的问答；系统通过混合 RAG 检索获取依据，通过 ReAct 自主调用检索、论文搜索和计算等工具，并使用 LangGraph 支持复杂任务的多 Agent 协作，同时具备跨会话记忆、调用链追踪和质量评测能力。

### 1.3 五条核心亮点

1. **多知识库管理**：支持用户创建、维护和指定多个科研知识库，通过独立 ChromaDB Collection 实现用户级和知识库级隔离。
2. **混合 RAG 检索**：对论文进行解析、切分和向量化，通过向量检索与 BM25 双路召回、RRF 融合和二阶段重排，提高专业术语与语义内容的检索稳定性。
3. **智能任务执行**：实现自研 ReAct 单 Agent 执行引擎，并基于 LangGraph 构建 Supervisor 多 Agent 编排，支持工具自主调用、角色分工和多步骤任务执行。
4. **分层记忆体系**：使用 Redis + SQLite 管理短期会话，使用 ChromaDB 保存跨会话长期事实，并提供 SQLite 降级召回。
5. **质量保障体系**：统一统计 Token 与模型成本，使用 qrels 评测 Recall、MRR、nDCG 和检索延迟，并通过 pytest 与 GitHub Actions 执行自动化回归。

---

## 2. 项目完整运行链路

```text
用户登录并选择知识库
        ↓
上传 PDF/TXT/Markdown 等科研资料
        ↓
文档解析、清洗、去重、切分
        ↓
Embedding 向量化
        ↓
写入 ChromaDB/FAISS，同时维护 BM25 索引
        ↓
用户提出问题
        ↓
查询改写（可选）
        ↓
向量检索 + BM25 双路召回
        ↓
按 doc_id 去重 + RRF 融合
        ↓
Reranker 二阶段重排 + Top K
        ↓
ReAct Agent 判断是否继续调用工具
        ↓
简单任务由单 Agent 完成；复杂任务可进入 LangGraph 多 Agent
        ↓
生成带来源的答案并流式返回
        ↓
保存会话、提取长期事实、记录 Token/Trace、执行质量评估
```

---

## 3. 文档上传与多知识库隔离

### 3.1 为什么需要多知识库

科研用户往往同时维护不同课题，例如“医学影像”“大模型”“实验记录”。如果全部资料混在一个库中，检索容易互相干扰。因此系统允许用户在问答时明确指定目标知识库。

### 3.2 如何隔离

每个用户、每个知识库使用独立的 ChromaDB Collection：

```text
默认库：kb_{username}
命名库：kb_{username}__{library_slug}
```

中文知识库名称会转换成合法的 ASCII 标识，真实中文显示名保存在 Collection metadata 中。请求进入知识库工具时，系统通过当前用户名和知识库名确定唯一 Collection，避免跨用户检索。

关键代码：

- Collection 命名规则：[user_kb.py](D:/科研助手agent/app/knowledge/user_kb.py:123)
- 用户知识库绑定 ChromaDB：[user_kb.py](D:/科研助手agent/app/knowledge/user_kb.py:267)
- 上传接口：[user_kb.py](D:/科研助手agent/app/api/routes/user_kb.py:88)
- 创建知识库：[user_kb.py](D:/科研助手agent/app/api/routes/user_kb.py:277)
- 检索时选择用户与知识库：[knowledge_search.py](D:/科研助手agent/app/tools/knowledge_search.py:88)

### 3.3 文档切分

长论文不能整体塞入模型，也不利于精确检索，因此需要拆成 Chunk。每个 Chunk 保存：

```text
doc_id：父文档标识
chunk_id：文档片段标识，例如 paper01#3
text：片段正文
metadata：来源、页码等信息
```

Chunk 太大时，检索结果包含大量无关上下文；太小时，语义可能被切断。`chunk_overlap` 让相邻 Chunk 保留部分重叠内容，减少边界信息丢失。

关键代码：[chunking.py](D:/科研助手agent/app/knowledge/chunking.py:29)、[chunking.py](D:/科研助手agent/app/knowledge/chunking.py:82)

---

## 4. Embedding、余弦相似度与向量库

### 4.1 Embedding 是什么

Embedding 将文本映射为数值向量。语义相近的文本在向量空间中的方向通常更接近：

```text
“医学影像分割方法” → [0.12, -0.35, 0.81, ...]
“医学图像如何分割” → [0.10, -0.31, 0.79, ...]
```

文档和查询必须使用同一个 Embedding 模型，否则向量不在同一语义空间中。

关键代码：[embeddings.py](D:/科研助手agent/app/knowledge/embeddings.py:73)

### 4.2 余弦相似度如何计算

```text
cos(A,B) = (A·B) / (||A|| × ||B||)
```

分数越大表示向量方向越接近。向量经过 L2 归一化后，长度都是 1，此时：

```text
余弦相似度 = 向量内积
```

Sentence-Transformers 和测试用 HashingEmbedding 都会归一化向量。需要注意：Ollama 路径当前直接接收模型返回值，项目没有再次归一化，因此必须确认具体 Ollama Embedding 模型是否输出归一化向量。

归一化代码：[embeddings.py](D:/科研助手agent/app/knowledge/embeddings.py:108)

### 4.3 FAISS 是什么

FAISS 是 Meta 开源的向量相似度检索库。项目使用 `IndexFlatIP` 保存向量并计算内积；在向量归一化的前提下，内积等价于余弦相似度。

```text
查询文本 → 查询向量 → FAISS 与文档向量批量比较 → 返回 Top K
```

`IndexFlatIP` 是精确检索：结果准确，但数据量极大时需要扫描全部向量。更大规模可切换 IVF 或 HNSW 等近似索引。

关键代码：[faiss.py](D:/科研助手agent/app/knowledge/stores/faiss.py:16)、[faiss.py](D:/科研助手agent/app/knowledge/stores/faiss.py:61)、[faiss.py](D:/科研助手agent/app/knowledge/stores/faiss.py:68)

### 4.4 FAISS 与 ChromaDB 的分工

| 存储 | 项目中的主要用途 | 特点 |
|---|---|---|
| FAISS | 公共知识库、本地评测 | 轻量、检索快，元数据需自行维护 |
| ChromaDB | 用户多知识库、长期事实 | 支持 Collection、元数据过滤和服务化 |
| Redis | 热会话和缓存 | 低延迟、支持 TTL，但不是最终持久化来源 |
| SQLite | 会话持久化、元数据、降级存储 | 零额外服务，适合单机，但并发能力有限 |

---

## 5. BM25 关键词检索

BM25 擅长精确匹配科研术语、缩写、编号和数字。它主要考虑：

1. **TF**：查询词在当前文档中出现多少次。
2. **IDF**：该词在整个语料中是否稀有；越稀有，区分能力越强。
3. **文档长度归一化**：避免长文档仅因为词多而获得虚高分。

简化公式：

```text
BM25(D,Q) = Σ IDF(q) × f(q,D)×(k1+1)
                        ─────────────────────────
                        f(q,D)+k1×(1-b+b×|D|/avgdl)
```

向量检索解决“语义相近”，BM25 解决“字面精确”，两者互补。

关键代码：[bm25.py](D:/科研助手agent/app/knowledge/bm25.py:23)、[bm25.py](D:/科研助手agent/app/knowledge/bm25.py:42)、[bm25.py](D:/科研助手agent/app/knowledge/bm25.py:57)

---

## 6. 混合 RAG：双路召回、RRF 与二阶段重排

### 6.1 为什么不能直接相加

向量相似度和 BM25 分数来自不同计算体系，数值范围和意义不同，直接相加会让某一路因为分值尺度更大而支配结果。

### 6.2 RRF 如何计算

RRF（Reciprocal Rank Fusion）只看各路排名，不直接比较原始分数：

```text
RRF(doc) = Σ 1 / (k + rank)
```

项目 `k=60`，代码排名从 0 开始，所以实际贡献为：

```text
1 / (60 + rank + 1)
```

文档 A 在向量检索第 1、BM25 第 2：

```text
RRF(A) = 1/61 + 1/62 ≈ 0.0325
```

文档 B 只在向量检索第 1：

```text
RRF(B) = 1/61 ≈ 0.0164
```

所以两路都认可的 A 通常排得更靠前。当前实现是等权 RRF；准确说法是“两路排名具有同等话语权”，而不是把原始分数按 50% + 50% 相加。

关键代码：[fusion.py](D:/科研助手agent/app/rag/fusion.py:10)

### 6.3 候选集如何确定

项目先扩大候选池：

```text
candidate_k = max(最终 top_k × 3, rerank_top_n, 最终 top_k)
```

如果最终返回 5 条，通常先让向量检索和 BM25 各取最多 15 条，再去重、融合和精排。

关键代码：[pipeline.py](D:/科研助手agent/app/rag/pipeline.py:144)

### 6.4 RRF 与 Reranker 的区别

```text
RRF：这篇文档在多个检索器中分别排第几？
Reranker：这篇文档的内容究竟是否适合当前问题？
```

RRF 负责低成本合并候选；Reranker 对候选做第二轮筛查。不能直接让 Reranker 处理整个知识库，因为逐个比较所有文档会带来很高延迟。

### 6.5 当前默认轻量重排

当前生产默认使用 `LexicalReranker`：

```text
重排分 = 0.7 × 归一化向量分 + 0.3 × 查询/文档词面重合数
```

它会重新强调向量相似度，并用关键词匹配纠偏。需要准确说明：当前公式使用候选的原始向量分数，没有把 `rrf_score` 纳入最终重排公式，因此 RRF 主要负责合并、去重和形成候选集。

关键代码：[rerank.py](D:/科研助手agent/app/knowledge/rerank.py:49)、[pipeline.py](D:/科研助手agent/app/rag/pipeline.py:183)

### 6.6 可选 Cross-Encoder

Cross-Encoder 把 `(query, document)` 同时送入模型，直接输出相关性分数：

```text
(问题, 文档A) → 0.42
(问题, 文档B) → 0.93
```

项目支持 `BAAI/bge-reranker-v2-m3`，并实现加载/推理失败后自动降级到 LexicalReranker。现有消融实验显示，当前 CPU 和语料下 Cross-Encoder 延迟约 18.5 秒/查询，中文指标还略低于轻量重排，因此保留为可插拔能力而非默认方案。

关键代码：[rerank.py](D:/科研助手agent/app/knowledge/rerank.py:70)、[rerank.py](D:/科研助手agent/app/knowledge/rerank.py:120)

---

## 7. ReAct 单 Agent 执行引擎

ReAct = Reasoning + Acting，核心是循环：

```text
理解问题 → 判断是否调用工具 → 执行工具 → 读取 Observation
        ↑                              ↓
        └──────── 继续推理 ────────────┘
                        ↓
                   输出最终答案
```

实际流程：

1. 写入用户消息，读取系统提示、会话历史和长期事实。
2. 大模型生成普通答案或 `tool_calls`。
3. 若存在多个工具调用，使用 `asyncio.gather` 并行执行。
4. 将 Assistant 的 tool call 和对应 ToolMessage 成对写回历史。
5. 重新调用模型；没有 tool call 时结束。

工程护栏包括：最大迭代次数、最大工具次数、Token 预算、Observation 截断、工具超时、可重试错误重试、重复调用拦截、连续失败熔断和达到限制后的强制收尾。

关键代码：

- ReAct 主循环：[agent.py](D:/科研助手agent/app/agent/agent.py:118)
- 最大迭代循环：[agent.py](D:/科研助手agent/app/agent/agent.py:164)
- 工具并行执行：[agent.py](D:/科研助手agent/app/agent/agent.py:231)
- 强制收尾：[agent.py](D:/科研助手agent/app/agent/agent.py:289)
- 工具超时/重试/重复拦截：[executor.py](D:/科研助手agent/app/tools/executor.py:54)

---

## 8. LangGraph 与多 Agent 协作

### 8.1 LangGraph 单 Agent 图

LangGraph 把 ReAct 循环显式建模成状态图：

```text
START → reason
          ├─ 有 tool_calls → tools → reason
          └─ 无 tool_calls → END
```

关键代码：[single_agent.py](D:/科研助手agent/app/graph/single_agent.py:48)

### 8.2 Supervisor 多 Agent

角色分工：

| 角色 | 职责与工具 |
|---|---|
| Researcher | 知识库、论文搜索、PDF 阅读、代码执行，负责收集事实 |
| Analyst | 计算器、引用、代码执行，负责数据分析和规范化 |
| Writer | 不调用外部工具，负责整合和写作 |
| Supervisor | 根据任务进展决定委派对象或结束流程 |

图结构：

```text
START → Supervisor → Researcher/Analyst/Writer → Supervisor → ... → END
```

多 Agent 不是所有问题都必需。简单任务使用单 Agent 更快、更便宜；只有需要角色分工和多阶段交付的复杂任务才值得使用多 Agent。

关键代码：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:55)、[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:108)、[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:266)

### 8.3 共享黑板、Inbox 与 Chain

`TeamState` 包含：

```text
messages：完整消息与工具调用历史
blackboard：各专家的最新阶段性成果
inbox：定向交付给目标专家的上游成果
chain：记录 A → B → C 的交接顺序
```

Blackboard 使用合并 Reducer：不同专家的结果保留，同一专家的新结果覆盖旧结果。Supervisor 读取黑板决定下一步，并把上游成果写入目标专家 Inbox，减少重复检索和计算。任务结束后，黑板成果可以沉淀到长期记忆。

关键代码：

- Reducer：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:68)
- TeamState：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:95)
- Supervisor 读取与交接：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:163)
- Worker 读取 Inbox 并写黑板：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:215)
- 黑板持久化：[multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:298)

当前需要注意：handoff 的 `_payload_` 在 Worker 拼接上下文时被跳过，主管附加指令存在未正确注入的风险；专家成果直传本身可以工作。这属于可继续修复的实现缺口。

---

## 9. 分层记忆体系

### 9.1 L1：短期会话记忆

保存用户消息、Agent 回答、tool calls、工具结果、标题和摘要。默认滑动窗口为最近 20 条，也可以根据 Token 预算截断，并保证 tool call 与 ToolMessage 不被拆开。

存储策略：

```text
写入：Redis 热数据 + SQLite 持久化兜底
读取：优先 Redis；未命中则从 SQLite 恢复并回填 Redis
```

Redis 数据可能因 TTL、内存淘汰、服务重启或持久化时间窗口消失，因此不能把 Redis 当作唯一可靠存储。双写方案提高了可恢复性，但当前不是严格分布式事务，只能称为最终可恢复或尽力一致。

关键代码：[conversation.py](D:/科研助手agent/app/memory/conversation.py:15)、[conversation.py](D:/科研助手agent/app/memory/conversation.py:95)、[persistence.py](D:/科研助手agent/app/memory/stores/persistence.py:23)、[persistence.py](D:/科研助手agent/app/memory/stores/persistence.py:157)

### 9.2 L2：长期事实记忆

回答结束后，大模型最多提取若干条对未来有价值的事实，例如用户研究方向、偏好、关键结论和使用过的方法。

```text
对话 → LLM 提取事实 → Embedding → ChromaDB memory_facts
```

新问题到来时，系统使用当前问题语义检索相关事实，默认取 3 条注入系统提示；如果 ChromaDB 不可用，则降级到 SQLite FTS5，仍未命中时使用 LIKE。

关键代码：[finalizer.py](D:/科研助手agent/app/agent/finalizer.py:101)、[longterm.py](D:/科研助手agent/app/memory/longterm.py:34)、[longterm.py](D:/科研助手agent/app/memory/longterm.py:164)、[longterm.py](D:/科研助手agent/app/memory/longterm.py:211)、[context.py](D:/科研助手agent/app/agent/context.py:66)

### 9.3 L3：会话索引

SQLite 保存 session_id、标题、摘要、创建时间和更新时间，用于历史会话列表、搜索和恢复。它属于结构化会话管理，不等同于语义长期记忆。

---

## 10. Token、Trace 与成本统计

模型 API 返回：

```text
prompt_tokens：输入消耗
completion_tokens：输出消耗
total_tokens：总消耗
```

项目在统一 LLM Provider 层采集真实 Usage，并由 `UsageLedger` 按模型累计请求量、输入/输出 Token 和估算成本：

```text
成本 = 输入Token/1000×输入单价 + 输出Token/1000×输出单价
```

Agent 内部的 Token 估算用于上下文预算护栏，不等同于 API 返回的真实计费 Token。流式输出改善等待体验，但通常不会降低 Token 消耗。

关键代码：[openai_provider.py](D:/科研助手agent/app/llm/openai_provider.py:140)、[usage.py](D:/科研助手agent/app/llm/usage.py:30)、[usage.py](D:/科研助手agent/app/llm/usage.py:68)、[admin.py](D:/科研助手agent/app/api/routes/admin.py:238)

---

## 11. 检索评测与指标计算

项目使用 qrels 保存“问题—相关文档”标准答案，再让生产 Retriever 返回 Top K，与标准答案比较。

### 11.1 Precision@K

```text
Precision@K = 前K条中的相关文档数 / K
```

回答“返回的结果准不准”。

### 11.2 Recall@K

```text
Recall@K = 前K条中的相关文档数 / 全部相关文档数
```

回答“应该找到的资料找全了多少”。

### 11.3 HitRate@K

前 K 条只要至少命中一条相关文档就是 1，否则为 0；多个问题再取平均。

### 11.4 MRR@K

单个问题：

```text
RR = 1 / 第一条相关文档的排名
```

多个问题取 RR 平均值。它关注第一条正确资料是否足够靠前。

### 11.5 nDCG@K

```text
DCG@K  = Σ (2^rel_i - 1) / log2(i + 1)
nDCG@K = DCG@K / IDCG@K
```

它既考虑相关程度，也惩罚相关文档排得太靠后；`1` 表示与理想排序一致。

### 11.6 平均检索延迟

```text
平均延迟 = 所有查询检索耗时总和 / 查询数量
```

通常包括查询向量化、双路召回、RRF 和重排，不包括大模型最终生成答案。

### 11.7 消融评测

固定语料、问题和 qrels，每次只改变一个模块：

```text
仅向量
向量 + BM25 + RRF
RRF + Lexical
RRF + Cross-Encoder
```

比较 Recall、MRR、nDCG 和延迟，判断新增模块是否真的带来收益，而不是只堆技术名词。

关键代码：[eval_retrieval.py](D:/科研助手agent/app/eval_retrieval.py:51)、[eval_retrieval.py](D:/科研助手agent/app/eval_retrieval.py:107)

### 11.8 Agent 端到端评测

项目将评测分为四层，避免只看检索分数：

- **检索层**：`Recall@K` 判断正确资料是否进入候选集，MRR/nDCG 补充衡量排序。
- **执行层**：`Tool Call Success Rate = 成功执行的工具调用数 / 全部工具调用数`，另算工具选择准确率。
- **任务层**：`Task Success Rate = 通过全部验收条件的用例数 / 总用例数`；验收包含无异常、答案非空、类型正确、工具与参数正确、关键内容命中及质量门槛。
- **答案层**：Answer Correctness 默认由 Exact Match 与 Token F1 加权计算；Groundedness 将答案拆成事实声明，统计能被检索上下文支持的比例。两者均可选 LLM-as-Judge，失败时自动降级为离线确定性算法。
- **性能层**：记录端到端平均延迟、P50 和 P95；P95 表示 95% 的请求延迟不超过该值。

关键代码：[answer_eval.py](D:/科研助手agent/app/evaluation/answer_eval.py:1)、[agent_eval.py](D:/科研助手agent/app/evaluation/agent_eval.py:45)、[eval_agent.py](D:/科研助手agent/scripts/eval_agent.py:22)

---

## 12. 自动化测试与 CI

默认离线测试覆盖：

- ReAct、LangGraph 与多 Agent 交接
- 工具超时、重试、重复调用和失败自修正
- BM25、RRF、Reranker 与 RAG Pipeline
- 多知识库和用户权限隔离
- Redis、SQLite、ChromaDB 与长期记忆
- Token 预算、上下文压缩、Usage 与 Trace
- API、WebSocket、会话管理和安全审计

当前套件共收集 300 项：本机验证结果为 `284 passed, 12 skipped, 4 deselected`。其中网络和 Chroma 集成测试默认排除或按环境跳过，避免外部服务波动影响离线 CI。新增功能应关注覆盖的关键路径，而不应只强调测试数量。

关键配置：[pyproject.toml](D:/科研助手agent/pyproject.toml:42)、[ci.yml](D:/科研助手agent/.github/workflows/ci.yml:39)

---

## 13. 十个高频面试问题与回答

### Q1：为什么向量检索和 BM25 要一起使用？

向量检索擅长语义泛化，BM25 擅长术语、缩写、数字和精确关键词。科研语料同时包含自然语言表达和大量专有名词，因此双路召回比单一路径更稳定。

### Q2：RRF 和 Reranker 有什么区别？

RRF 根据多路排行榜中的名次合并候选，不理解内容；Reranker 对候选做第二轮相关性判断。前者解决“名单怎么合并”，后者解决“候选中谁更适合问题”。

### Q3：为什么不让 Reranker 检查整个知识库？

Cross-Encoder 等重排器需要逐个计算问题—文档对，直接处理全库成本太高。先召回少量候选，再精排，能够平衡效果与延迟。

### Q4：ReAct 如何避免无限循环？

项目限制最大迭代次数和工具次数，并设置 Token 预算、工具超时、重复调用拦截和连续失败熔断。达到限制后要求模型根据已有信息强制收尾。

### Q5：为什么需要多 Agent？

单 Agent 适合简单任务；复杂科研任务可能需要检索、计算和写作等不同角色。多 Agent 能隔离提示词和工具权限，但增加调用次数、延迟和成本，因此应按任务复杂度选择，而不是全部使用。

### Q6：共享黑板和 Inbox 有什么区别？

Blackboard 是全体 Agent 可见的公共成果区；Inbox 是定向交给某个 Agent 的任务包。Supervisor 根据黑板做决策，再用 Inbox 把上游成品明确交给下游，减少重复工作。

### Q7：Redis 为什么还要配 SQLite？

Redis 读取快，但 Key 可能因 TTL、淘汰或重启消失。项目将消息双写 SQLite；Redis 未命中时从 SQLite 恢复并回填，以兼顾性能和可靠性。但当前双写不是严格事务。

### Q8：FAISS 和 ChromaDB 为什么同时存在？

FAISS 是轻量向量检索库，适合本地精确搜索和评测；ChromaDB 是带 Collection 和元数据能力的向量数据库，更适合用户多知识库和长期事实。项目按场景选择后端。

### Q9：如何证明 RAG 优化有效？

使用固定 qrels 做消融实验，对比 Recall、MRR、nDCG 和延迟，并在 CI 中设置检索回归门禁。只有指标真实提升，才能声称优化有效。

### Q10：项目如何进一步生产化？

将 SQLite 迁移到 PostgreSQL；使用消息队列和 Outbox 改善双写一致性；扩充真实 qrels；对多 Agent 增加复杂度路由和成本预算；对长期事实增加置信度、版本和过期机制；补充监控、告警、限流和审计。

---

## 14. 当前不足与表达边界

1. **不要说 Cross-Encoder 已作为默认生产方案**：它是可插拔能力，当前默认是 LexicalReranker。
2. **不要说 Redis + SQLite 是强一致**：当前是双写和读取恢复，没有分布式事务。
3. **不要说 RAG 消除了幻觉**：RAG 只能提供证据并降低幻觉，仍需引用校验和回答评测。
4. **不要把多 Agent 描述成所有任务的最优解**：它提高职责清晰度，但增加延迟和成本。
5. **不要只强调测试数量**：应强调核心链路、异常路径、质量指标和回归门禁。
6. **评测数据规模仍有限**：现有结果适合项目内对比，不能直接代表所有真实科研场景。
7. **SQLite 更适合单机演示**：高并发生产环境应考虑 PostgreSQL 等服务化数据库。
8. **长期记忆可能过期或错误**：后续应增加事实置信度、更新时间、来源和删除/修订机制。

---

## 15. 关键代码索引

| 模块 | 关键入口 |
|---|---|
| 应用入口 | [main.py](D:/科研助手agent/app/main.py:1) |
| 用户知识库接口 | [user_kb.py](D:/科研助手agent/app/api/routes/user_kb.py:88) |
| 知识库隔离 | [user_kb.py](D:/科研助手agent/app/knowledge/user_kb.py:123) |
| 文档切分 | [chunking.py](D:/科研助手agent/app/knowledge/chunking.py:29) |
| Embedding | [embeddings.py](D:/科研助手agent/app/knowledge/embeddings.py:73) |
| FAISS | [faiss.py](D:/科研助手agent/app/knowledge/stores/faiss.py:16) |
| BM25 | [bm25.py](D:/科研助手agent/app/knowledge/bm25.py:23) |
| RRF | [fusion.py](D:/科研助手agent/app/rag/fusion.py:10) |
| RAG Pipeline | [pipeline.py](D:/科研助手agent/app/rag/pipeline.py:131) |
| Reranker | [rerank.py](D:/科研助手agent/app/knowledge/rerank.py:49) |
| ReAct Agent | [agent.py](D:/科研助手agent/app/agent/agent.py:118) |
| ToolExecutor | [executor.py](D:/科研助手agent/app/tools/executor.py:54) |
| LangGraph 单 Agent | [single_agent.py](D:/科研助手agent/app/graph/single_agent.py:48) |
| 多 Agent Supervisor | [multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:108) |
| 共享黑板 | [multi_agent.py](D:/科研助手agent/app/graph/multi_agent.py:95) |
| 短期记忆 | [conversation.py](D:/科研助手agent/app/memory/conversation.py:15) |
| Redis/SQLite 会话存储 | [persistence.py](D:/科研助手agent/app/memory/stores/persistence.py:23) |
| 长期记忆 | [longterm.py](D:/科研助手agent/app/memory/longterm.py:34) |
| 长期事实注入 | [context.py](D:/科研助手agent/app/agent/context.py:66) |
| Token/成本账本 | [usage.py](D:/科研助手agent/app/llm/usage.py:30) |
| 检索评测 | [eval_retrieval.py](D:/科研助手agent/app/eval_retrieval.py:51) |
| 测试配置 | [pyproject.toml](D:/科研助手agent/pyproject.toml:42) |
| CI | [ci.yml](D:/科研助手agent/.github/workflows/ci.yml:39) |

---

## 16. 一分钟项目介绍

> 这是一个面向科研调研和知识问答的 Agent 系统。用户可以创建多个独立知识库并上传论文，系统对文档进行解析、切分和向量化，通过向量检索与 BM25 双路召回，再使用 RRF 融合和二阶段重排筛选证据。执行层实现了自研 ReAct 单 Agent，可自主调用知识库、论文搜索和计算等工具；复杂任务则通过 LangGraph Supervisor 分配给 Researcher、Analyst 和 Writer，并使用共享黑板传递阶段成果。系统还使用 Redis、SQLite 和 ChromaDB 构建分层记忆，配套 Token 成本统计、检索指标、调用链追踪与自动化测试，从而形成从知识入库、任务执行到质量保障的完整闭环。
