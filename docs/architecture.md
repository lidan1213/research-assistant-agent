# 项目架构与收敛约定

## 系统主链路

当前正式主链路是：

`API -> app.api.deps -> ResearchAgent -> ToolRegistry / ConversationMemory / RetrievalService -> LLM Gateway`

`app.graph` 中的 LangGraph 单 Agent 和多 Agent 是实验性运行策略。新增业务能力应优先接入正式主链路，不应在 `app.graph` 中复制工具、记忆或检索实现。

## 模块职责

- `app/api`：HTTP、SSE 和 WebSocket 适配，不承载持久化与复杂业务规则。
- `app/services/sessions.py`：用户会话列表、归属校验、管理、检索和导出数据。
- `app/services/sharing.py`：只读分享链接的签名、过期校验和共享会话读取。
- `app/services/admin_metrics.py`：管理员成本与检索质量的纯数据聚合。
- `app/services/admin_students.py`：学生账号、密码和会话管理。
- `app/agent`：Agent 编排、规划、上下文和短期会话记忆。
- `app/agent/events.py`：所有 Agent 运行策略共享的流式事件模型。
- `app/agent/finalizer.py`：回答后的会话摘要、质量自评和长期事实提取。
- `app/llm`：模型供应商、路由、结构化输出、用量和调用追踪。
- `app/tools`：工具协议、注册和执行。
- `app/rag`：新的检索流水线、融合逻辑和统一结果类型。
- `app/knowledge`：现有索引、嵌入和知识库实现，后续逐步迁入 `app/rag`。
- `app/knowledge/stores`：向量存储协议、独立后端实现与严格工厂；旧 `vectorstore.py` 仅保留兼容导出。
- `app/memory`：短期会话服务、内存/SQLite/Redis 存储和跨会话长期记忆。
- `app/graph`：可选 LangGraph 策略，不作为领域能力的归属位置。
- `app/evaluation`：离线质量评测及评测结果存储。

## 依赖方向

```text
api -> agent/runtime -> llm, tools, rag, memory
tools -> knowledge/rag
graph -> llm, tools
```

底层模块不得反向导入 API 路由。路由通过 `app.api.deps` 获取服务，避免自行创建模型客户端、数据库连接或 Agent。

## 数据与生成物

源码目录中不保存运行日志、数据库、向量索引和备份。当前 `data/` 保留为兼容路径；新部署应将对应路径配置到项目外部持久卷。临时测试产物统一写入 `.test-tmp/`。

## 后续重构顺序

1. Agent API 已通过 `AgentRuntime` 协议与具体实现解耦。
2. `RetrievalPipeline` 现已拥有入库和检索；`Retriever` 保留为兼容门面。
3. FAISS、SQLite、本地 Chroma 与远程 Chroma 均已迁入独立后端模块。
4. `ConversationMemory` 与内存、SQLite、Redis 会话仓储均已迁入 `app.memory`；旧 `agent.memory` 仅保留兼容导出。
5. `chat.py` 会话管理与分享、`admin.py` 的统计和学生管理已迁入 service 层。
6. 完成兼容重构后，再将源码迁入 `src/research_assistant/`，避免包名 `app` 冲突。

每一步都应保持已有 API 路径兼容，并先补充回归测试再移动实现。
