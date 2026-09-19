"""应用配置中心：基于 pydantic-settings 读取环境变量 / .env。

设计要点：
1. 所有可配置项集中在此，运行时通过 `get_settings()` 单例访问。
2. 敏感字段（API Key 等）只从环境读取，不落盘到代码。
3. 使用 `BaseModel` 子模型对配置分组，方便解耦与测试。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 60
    # 模型上下文窗口（token）。用于计算对话 token 预算：
    # budget = max_context_tokens × 0.6（预留工具调用 + 输出空间）
    max_context_tokens: int = 128000
    # ---- 多模型路由（ModelRouter）----
    # 辅助模型：用于内部低价值调用（对话摘要/事实提取/查询改写/强制收尾等），
    # 通常配置为更便宜的模型以降低成本；留空 = 与主模型相同。
    # provider 留空 = 与主模型相同供应商；base_url/api_key 留空 = 沿用主配置。
    aux_provider: str = ""
    aux_model: str = ""
    aux_base_url: str = ""
    aux_api_key: str = ""
    # 用户可自选的模型列表（逗号分隔）。空 = 不限制（前端从供应商 /v1/models 拉取）
    available_models: str = ""


class MemorySettings(BaseSettings):
    redis_url: str = ""           # 留空则使用本地内存
    ttl: int = 86400
    # 会话记忆后端：memory（进程内，重启丢失）| sqlite（落盘，可恢复）| redis（需填 redis_url）
    backend: str = "memory"
    sqlite_path: str = "./data/memory.db"      # 会话记忆落盘路径
    longterm_path: str = "./data/longterm.db"  # 长期记忆（跨会话事实/结论）落盘路径


class ToolSettings(BaseSettings):
    # bing_html = 免费抓取 Bing 网页结果（国内直连可用，无需 key，默认）
    # duckduckgo = 免 key 但国内被墙；serpapi / bing = 需对应 API Key
    search_provider: Literal["duckduckgo", "serpapi", "bing", "bing_html"] = "bing_html"
    serpapi_key: str = ""
    bing_subscription_key: str = ""
    code_exec_timeout: int = 10
    # 代码执行沙箱模式：
    #   subprocess = 本机子进程执行（默认，仅适合本地/受信任环境）
    #   docker = Docker 容器隔离执行（--network none --memory 限制，生产推荐）
    #   auto = 自动检测 docker 可用则用，否则回退 subprocess
    code_exec_sandbox: Literal["subprocess", "docker", "auto"] = "subprocess"
    # Docker 执行时使用的镜像（需含 python 解释器；无 docker 时忽略）
    code_exec_docker_image: str = "python:3.11-slim"
    code_exec_memory_mb: int = 512   # 容器内存上限（MB）
    code_exec_cpus: float = 1.0      # 容器 CPU 上限


class KnowledgeSettings(BaseSettings):
    # 嵌入模型：
    # - 默认本地 sentence-transformers 语义嵌入 "BAAI/bge-small-zh-v1.5"（首次使用下载权重，依赖 torch）。
    # - 想用本地 Ollama 生成嵌入时，改为 "ollama://<模型名>"，
    #   可选携带 host："ollama://<host:port>/<模型名>"（默认 http://localhost:11434）。
    #   需本地已运行 ollama serve 并拉取对应模型（如 `ollama pull nomic-embed-text`）。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 向量库后端: faiss（本地 .faiss 文件，默认）| chroma（本地 ChromaDB 数据库）
    vector_store: str = "faiss"
    vector_top_k: int = 5
    rerank_top_k: int = 3
    # 文档分块：默认关闭以保持现有单文档基线；扩展语料可在 .env 开启。
    chunking_enabled: bool = False
    chunk_size: int = 400
    chunk_overlap: int = 80
    # 当前语料消融实验中 lexical 的质量/延迟更优；Cross-Encoder 保留为可选精排。
    reranker_kind: str = "lexical"  # lexical | cross-encoder | none
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_min_score: float = 0.35
    low_confidence_score: float = 0.45
    # 重排阶段的默认权重：科研文档优先保留语义相似度，词面匹配用于纠偏。
    # 通过 .env 可做离线消融实验，不需要改代码。
    rerank_vector_weight: float = 0.7
    rerank_lexical_weight: float = 0.3
    knowledge_dir: str = "./data/knowledge"


class AgentSettings(BaseSettings):
    """Agent 运行护栏（防止长链路失控 / 上下文爆炸）。"""

    max_iterations: int = 8          # ReAct 最大推理-行动轮数
    max_tool_calls: int = 15         # 全部工具调用总次数上限（防联网搜索死循环）
    # 近似上下文 token 预算，超出后强制收尾。
    # 默认 0 = 自动按模型窗口计算：max_context_tokens × 0.6；
    # 显式设置 >0 时优先用该值（覆盖自动计算）。
    token_budget: int = 0
    max_obs_chars: int = 4000        # 单个工具 observation 超长截断阈值
    tool_max_retries: int = 1        # 工具执行失败后的自我修正重试次数
    tool_timeout: int = 20            # 单次工具调用超时，超时后快速回写失败并降级
    llm_recovery_timeout: int = 30    # 流式调用失败后，兜底生成的总超时（仅尝试一次）
    no_progress_limit: int = 2        # 工具返回重复结果达到次数后提前收尾
    recovery_budget: int = 2          # 无进展后最多切换两条替代检索路径
    # 历史压缩阈值：历史消息估算 token 超过 token_budget × 该比例时，
    # 把最早的对话轮次压缩为摘要（保留最近原文），防止长对话上下文膨胀。
    # 0 = 关闭历史压缩（仅靠 token 预算硬截断）。
    history_compress_ratio: float = 0.55
    # 压缩后保留的最近消息条数（原文保留，更早的压成摘要）
    history_keep_msgs: int = 16


class SkillSettings(BaseSettings):
    """本地 Agent Skills：仅加载项目内 SKILL.md，不执行任意 Skill 脚本。"""

    directory: str = "./skills"
    state_path: str = "./data/skills_state.json"


class Settings(BaseSettings):
    """顶层配置，聚合各子模块配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ResearchAssistantAgent"
    app_env: Literal["development", "production"] = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "./data/logs"  # 日志文件目录（日切轮转，保留 14 天）

    # 逗号分隔的浏览器来源。生产环境应通过环境变量显式配置。
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    api_keys: str = ""  # 逗号分隔；为空表示不鉴权
    # MCP stdio 服务器 JSON 数组；每项支持 name/command/args/env。
    mcp_servers: str = ""

    # 登录 Token 签名密钥（HMAC）。生产环境务必通过 .env 覆盖为强随机值
    auth_secret: str = "research-assistant-dev-secret-change-me"

    llm: LLMSettings = LLMSettings()
    memory: MemorySettings = MemorySettings()
    tools: ToolSettings = ToolSettings()
    knowledge: KnowledgeSettings = KnowledgeSettings()
    agent: AgentSettings = AgentSettings()
    skills: SkillSettings = SkillSettings()

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """返回全局配置单例。"""
    return Settings()
