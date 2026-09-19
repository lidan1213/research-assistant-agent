# 科研助手 Agent（Research Assistant Agent）

> 基于 **FastAPI** 的可演示、可评测、可离线运行的科研智能体框架。
> 以「单 Agent（ReAct）」与「多 Agent（LangGraph Supervisor）」两种模式协同，叠加本地知识库 RAG 与检索/答案质量评测。

---

## 0. 一句话定位

一个把 **推理编排（ReAct / 多 Agent）**、**本地知识库检索（混合检索 + 重排）**、**质量评测（IR 指标 + LLM-as-Judge）** 串成闭环的科研助手后端，前端 `/demo` 可实时观看推理全过程。Embedding 默认走本地 **Ollama**，推理 LLM 走 OpenAI 兼容协议（可换任意模型）。

---

## 1. 架构总览（模块职责）

```
app/
├── config.py            # 统一配置（pydantic-settings），LLM / 知识库 / Agent 护栏 / 记忆 / 工具
├── core/logging.py      # 日志
├── llm/
│   ├── base.py          # BaseLLM 抽象
│   ├── openai_provider.py  # AsyncOpenAI 实现（OpenAI 兼容协议）
│   └── usage.py         # 全局 UsageLedger：按模型统计 token 与成本
├── tools/
│   ├── registry.py      # 工具基类 + 装饰器 + JSON Schema 自动生成 + 注册表
│   └── *.py             # arxiv_search / pdf_reader（真实可用）/ web_search / calculator / citation / code_executor（沙箱隔离待做）
├── agent/
│   └── agent.py         # 手写 ReAct 推理循环 + 护栏（token 预算 / 观察截断 / 失败重试）
├── knowledge/           # RAG 检索链路
│   ├── embeddings.py    # EmbeddingModel：sentence-transformers / Ollama / Hashing 可切换
│   ├── chunking.py      # 递归分块 + overlap
│   ├── bm25.py          # 纯 Python BM25 稀疏索引
│   ├── rerank.py        # Lexical / CrossEncoder / Identity 重排器
│   ├── vectorstore.py   # FAISSVectorStore（clear 容错）
│   └── retriever.py     # 稠密+稀疏召回 → RRF 融合 → 重排
├── graph/               # LangGraph 编排
│   ├── langchain_tools.py  # 工具桥 + make_handoff（支持 payload）
│   ├── single_agent.py     # 单 Agent StateGraph
│   └── multi_agent.py      # 多 Agent：TeamState + 主管 + 专家 + 共享黑板 + 链路直传
├── evaluation/          # 评测体系
│   ├── metrics.py       # Recall / Precision / HitRate / MRR / nDCG / MAP / F1 @k
│   ├── dataset.py       # EvalDataset（JSON / TSV 解析）
│   ├── engine.py        # RetrievalEvaluator + 导出(CSV/Markdown/JSON) + 多配置对比
│   ├── answer_eval.py   # LLM-as-Judge：忠实度 / 答案相关性 / 上下文相关性
│   ├── sample_data.py   # 示例语料 + qrels
│   └── store.py         # 进程内评测运行存储（多配置横向对比）
├── api/
│   ├── deps.py          # get_llm / get_retriever
│   ├── routes/          # ws(chat) / graph / evaluation / knowledge
│   └── main.py          # create_app：CORS + StaticFiles(/demo) + lifespan
└── schemas/             # 请求/响应模型
frontend/index.html      # 零构建单文件演示页（对话演示 + 评测中心）
scripts/knowledge_eval.py# CLI：build / eval（--embedder ollama）
tests/                   # 300 项 pytest；网络/Chroma 集成测试使用 marker 显式运行
Dockerfile / docker-compose.yml / .github/workflows/ci.yml
.env                     # KNOWLEDGE__EMBEDDING_MODEL=ollama://nomic-embed-text 等
```

---

## 2. 两种运行模式（核心卖点）

| 维度 | 单 Agent（ReAct） | 多 Agent（Supervisor） |
|---|---|---|
| 结构 | 一个大脑自循环：思考→行动→观察→反思 | 主管拆解任务，委派 Researcher / Analyst / Writer 专家 |
| 协作机制 | 无 | 共享黑板（Blackboard）+ 定向 handoff + **链路直传 A→B→C** |
| 适用 | 单领域、步骤明确的任务 | 复杂多子任务（综述、对比、成稿） |
| 编排框架 | 手写 ReAct 循环 | LangGraph `StateGraph`（reason↔tools 有环） |
| 接口 | `WS /api/ws/chat` | `POST /api/graph/multiagent/run/stream`（SSE） |

两者**共用同一套工具与知识库**，差异只在"是否拆分专家 + 协作机制"。演示页顶部有「模式区别」对比条，可一键切换并实时看黑板/链路面板。

---

## 3. 数据流

**RAG 检索链路**
```
查询 → (可选查询改写) → 向量召回(FAISS) + BM25 召回 → RRF 融合 → 重排(默认 Lexical，可选 CrossEncoder) → 截断 top_k → 拼装上下文
```
Embedding 默认 `ollama://nomic-embed-text`（本地，768 维）；可换 sentence-transformers 或 Hashing（测试/CU）。

**Agent 推理链路**
```
用户问题 → 历史记忆 → LLM 决策(ReAct / Supervisor) → 选择工具 → 执行工具(ainvoke) → 观察(截断) → 反思 → … → 最终答案
```
护栏：每轮估算 token 预算，超限强制收尾；observation 超长截断；工具异常按 `tool_max_retries` 重试并回写 LLM 自修正。

**评测链路**
```
语料/查询 → 检索 → 比对 qrels → IR 指标(Recall/MRR/nDCG…)
答案 + 上下文 → LLM-as-Judge → 忠实度/相关性
多份运行 → store → build_comparison → 对比表/曲线图(前端) → 导出 CSV/Markdown/JSON
```

**记忆链路（跨会话持久化）**
```
[写入] 会话消息 ──→ L1 会话记忆(SQLite 落盘)
      多 Agent 黑板/链路成品 ──→ 汇总「本次研究结论」──→ L2 长期事实(facts 表)
                                                  └──→ L3 会话索引(sessions 表)
[读取] 新问题 ──→ 检索 L2(FTS5 全文/ LIKE 回退) + 拉取 L1(最近历史)
              ──→ 拼成上下文注入 LLM，使助手具备「跨会话记忆」
```
- `app/memory/conversation.py`：会话滑动窗口与 tool 消息配对。
- `app/memory/stores/`：`InMemoryStore` / `RedisStore` / **`SQLiteStore`** 独立后端。
- `app/memory/longterm.py`：`LongTermMemory` 基于 SQLite，零依赖、重启不丢；优先 FTS5 全文索引，不支持时自动回退 LIKE。
- `app/memory/manager.py`：`MemoryManager` 把三层串成一条链路，并暴露进程内单例 `get_memory_manager()`。
- `app/graph/multi_agent.py`：多 Agent 跑完时若带 `session_id`，自动把黑板/链路成品沉淀进长期记忆。

---

## 4. 快速开始

> ⚠️ **环境注意（避坑）**：本项目包名是顶层 `app`。若本机另有一个同名 `app` 包被 editable 安装进同一个解释器（例如另一项目的 `backend/app`），从子目录或共享解释器启动时可能误加载到别人的 `app` 而报错。本项目已通过 `conftest.py` + `run.py` 把项目根目录强制插到 `sys.path` 最前来规避。**推荐为本项目建独立虚拟环境**，一劳永逸：

```bash
# 1) 建并激活独立 venv（务必用项目自带的，避免共享解释器抢包）
python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate

# 2) 装依赖（也可 pip install -e . 把本项目的 app 注册进去）
pip install -e ".[dev,local-embedding]"
#    注：sentence-transformers（依赖 torch）仅默认本地 BGE embedding 需要；
#        当前配置用 Ollama embedding，无需 torch，可跳过以节省时间。

# 3) 本地 Embedding（默认已开启 Ollama）
ollama serve                 # 确保本机 11434 在跑
ollama pull nomic-embed-text # 768 维 embedding 模型

# 4) 推理 LLM：.env 已配好 DeepSeek（LLM__PROVIDER=deepseek + 已填 key）
#    若要换 gpt-4o-mini / Qwen / 本地 Ollama，改 LLM__BASE_URL + LLM__MODEL 即可。

# 5) 启动（推荐用 run.py，自动锁定本项目 app 包）
python run.py                 # 等价于 uvicorn app.main:app --reload
#    演示页： http://127.0.0.1:8000/demo

# 6) 离线检索评测
python scripts/knowledge_eval.py build --embedder ollama
python scripts/knowledge_eval.py eval  --embedder ollama --k 1 3 5
#    示例结果：recall@1=0.833 / MRR=0.833 / nDCG@1=0.833

# 7) 测试
pytest                        # 全量收集 300 项；当前离线结果 284 passed / 12 skipped / 4 deselected
```

---

## 5. 评测体系（质量闭环）

- **检索指标**：Recall / Precision / HitRate / MRR / nDCG / MAP / F1 @k（纯函数，见 `metrics.py`）。
- **Agent 端到端指标**：Task Success Rate、Answer Correctness、Groundedness、Tool Call Success Rate 与 P50/P95 Latency；任务成功按用例验收条件严格判定。
- **答案级评测**：默认使用可离线复现的 Exact Match / Token F1 / 声明证据覆盖，可选 LLM-as-Judge；Judge 失败时自动降级。
- **分层评测**：检索层使用 Recall / Precision / HitRate / MRR / nDCG / MAP / F1 @K，执行层使用工具与任务指标，结果层使用正确性与忠实度。
- **多配置横向对比**：保存命名运行（可开关 混合检索 / 重排），`GET /runs` + `POST /compare` 生成「配置×指标」对比矩阵，前端渲染分组柱状图。
- **结果导出**：`POST /evaluation/export` 支持 Markdown / CSV / JSON 附件下载。
- **回归门禁**：CI 中 `test_retrieval_regression` 断言 recall@5 ≥ 0.8，防止检索退化。

---

## 6. 工程化

- **Docker / Compose**：服务化打包。
- **CI（GitHub Actions）**：pytest + 检索回归门禁。
- **成本可观测**：`usage.py` 按模型累计 token 与成本。
- **前端零构建**：`frontend/index.html` 单文件，Tab（对话演示 / 评测中心）+ 实时事件流 + SVG 图表（无外部 CDN）。

---

## 7. 已实现亮点清单

1. 单 Agent 手写 ReAct + 多 Agent LangGraph Supervisor 双轨。
2. 共享黑板 + 定向 handoff + 链路直传（A→B→C）的多 Agent 协作。
3. 混合检索（向量 + BM25 + RRF）+ 可插拔重排。
4. Embedding 本地化（Ollama `nomic-embed-text`），知识库向量化离线零费用。
5. 检索 IR 指标 + 答案级 LLM-as-Judge 双评测 + 网页上传 qrels。
6. 多配置横向对比 + 评测结果导出（CSV/Markdown/JSON）。
7. Agent 护栏（token 预算 / 观察截断 / 失败自修正）+ 成本账本。
8. 流式演示页（WebSocket/SSE）可视化推理与委派全过程。
9. 记忆链路：会话记忆 SQLite 落盘 + 长期事实/会话索引跨会话持久化（FTS5 全文召回，零额外依赖）。
10. 真实可用的科研工具链：`arxiv_search`（联网检索 + 可选下载）↔ `pdf_reader`（本地/URL 读取 + 加密处理 + 结构化输出），形成「检索 → 精读」闭环，均带离线/联网测试。
11. 论文写作成品可导出 Markdown、Word（DOCX）和 PDF，保留标题层级、列表、中文排版与页码。
12. 本地 Agent Skills：自动扫描 `skills/*/SKILL.md`，支持触发词匹配、手动选择、管理员启停/重载，并将工作流安全注入 ReAct、Plan-and-Execute 和多 Agent。
13. MCP 工具生态：已接入官方 Filesystem MCP（限定 `data/`）与 Zotero MCP，外部工具会被动态注册到 Agent，可直接参与任务规划和执行。
14. 证据冲突感知：跨文档抽取观点、指标、数值和实验条件，区分同条件冲突与条件依赖差异；结合来源类型、同行评审、样本量及撤稿状态进行可审计的暂定裁决，无法裁决时并列呈现并保留引用。
15. 高频动态更新：SQLite 持久化文档版本状态，新版本完成索引验证后再原子激活，失败自动保留旧版本；知识库 revision 进入检索缓存键，并维护 ChromaDB/BM25 删除一致性和版本来源追溯。

### 本地 Skills

项目启动时从 `./skills` 加载 Skill。每个 Skill 是一个带 YAML front matter 的独立目录：

```text
skills/
└── literature-review/
    └── SKILL.md
```

```markdown
---
name: literature-review
description: 系统完成文献检索、筛选、证据整理与综述成稿
triggers: [文献综述, 研究现状, literature review]
tools: [knowledge_search, arxiv_search, web_search, pdf_reader]
preferred_mode: plan
---

这里填写需要 Agent 遵循的工作流指令。
```

- 普通用户可在对话框选择“自动匹配”或手动指定 Skill；
- 管理员可在“Skills”页面启停并重新扫描目录；
- `GET /api/skills` 查看列表，`POST /api/skills/match` 调试匹配；
- Skill 只注入指令并选择现有注册工具，不会执行 Skill 目录中的任意脚本；
- 默认附带 `literature-review`、`paper-review`、`experiment-design` 三个科研 Skill。

### MCP 工具接入

项目通过 `.env` 中的 `MCP_SERVERS` 启动 stdio MCP Server，并把工具注册为
`mcp_{server}_{tool}`。当前配置：

- **Filesystem MCP**：允许 Agent 读取和整理 `D:\科研助手agent\data`，不能越过该目录；首次启动会通过 `npx` 下载官方 Server。
- **Zotero MCP 0.6.4**：安装在 `var/mcp/zotero`，提供文献检索、元数据、全文、笔记、标注、集合与参考文献导出等工具。

Zotero 使用本地模式，无需注册账号或 API Key，但需要：

1. 安装并启动 Zotero 7；
2. 在“设置 → 高级 → API”中启用本地 API；
3. 保持 Zotero 运行后重启本项目。

若改用 Zotero Web API，则在 Zotero MCP 的 `env` 中配置
`ZOTERO_API_KEY`、`ZOTERO_LIBRARY_ID`，并移除或关闭 `ZOTERO_LOCAL`。

### 文档版本与频繁更新

- 同名文档内容变化时创建 `pending → indexing → active` 新版本；索引失败转为 `failed`，旧版本仍可检索；
- 激活成功后旧版本转为 `superseded`，再从 ChromaDB 与 BM25 派生索引清理；
- 每次激活、删除或清空都会递增知识库 `revision`，检索缓存键包含 revision，旧缓存自然失效；
- 检索结果携带 `logical_doc_id`、`version_id`、`version_number`，回答来源可以追溯到具体文档版本；
- `GET /api/userkb/versions?source=文件名&kb_name=库名` 查看版本状态、失败原因和当前 revision；
- 版本状态保存在 `data/document_versions.db`，可用环境变量 `KB_VERSION_DB` 覆盖路径。

---

## 8. 已知局限 / 后续方向

- ~~推理 LLM 尚未用真实 key 全链路跑通~~ → 已接入 **DeepSeek**（`.env` 已填 `LLM__PROVIDER=deepseek` 与 key），已实测可用；如需完全离线可把推理也切到本地 Ollama。
- ~~记忆为内存态、重启即丢~~ → 已落地 **记忆链路**（`SQLiteStore` 会话落盘 + `LongTermMemory` 长期事实/会话索引，FTS5 召回），详见「数据流-记忆链路」一节。
- 工具：~~多为占位/桩实现~~ → `arxiv_search`（联网检索 + 可选下载 PDF）、`pdf_reader`（本地/URL 读取、加密处理、结构化输出）已**真实可用并补了测试**；`web_search`/`calculator`/`citation` 可用；仅 `code_executor` 仍需真实沙箱隔离（暂为桩）。
- 会话记忆支持内存、SQLite 与 Redis+SQLite 双写；向量存储支持 FAISS、SQLite、本地/远程 Chroma。
- 评测样本较小（6 条 qrels），代表性有限。
- API 已实现登录鉴权、用户级资源隔离、管理员权限与聊天固定窗口限流；生产环境仍需配置强随机 `AUTH_SECRET` 和受限 CORS 来源。

> 附：本机 Ollama 安装与下载进度监控脚本在 `D:\ollama\monitor.py`（独立小工具，非项目依赖）。
