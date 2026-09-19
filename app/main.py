"""FastAPI 应用工厂：组装路由、中间件、异常处理、生命周期。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin, agent, auth, chat, graph, kg, knowledge, llm, notes, plan, skills, tools, user_kb, ws, writing
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging

logger = get_logger("bootstrap")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("科研助手 Agent 启动")
    # 将 Embedding/全局知识库的冷启动成本前移到 readiness 之前，
    # 避免第一批用户请求卡在模型权重加载。预热失败不影响服务启动。
    try:
        from app.tools.knowledge_search import _get_global_retriever

        await asyncio.wait_for(
            _get_global_retriever().retrieve("RAG", top_k=1), timeout=30
        )
        logger.info("知识库与 Embedding 预热完成")
    except Exception as exc:  # noqa: BLE001
        logger.warning("知识库预热失败，将在首次请求时重试: %s", exc)
    yield
    logger.info("科研助手 Agent 关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    if settings.debug:
        setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="基于 FastAPI 的科研助手 Agent 框架（ReAct + RAG + Tool Calling + LangGraph 多 Agent）",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS（按需收紧）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 统一异常处理
    register_exception_handlers(app)

    # 路由挂载
    api_prefix = "/api"
    app.include_router(auth.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(user_kb.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(agent.router, prefix=api_prefix)
    app.include_router(graph.router, prefix=api_prefix)
    app.include_router(plan.router, prefix=api_prefix)
    app.include_router(tools.router, prefix=api_prefix)
    app.include_router(notes.router, prefix=api_prefix)
    app.include_router(knowledge.router, prefix=api_prefix)
    app.include_router(llm.router, prefix=api_prefix)
    app.include_router(kg.router, prefix=api_prefix)
    app.include_router(ws.router, prefix=api_prefix)
    app.include_router(writing.router, prefix=api_prefix)
    app.include_router(skills.router, prefix=api_prefix)
    # 分享视图：独立无鉴权 router（公开只读，仅签名校验；叠加 api_prefix 得 /api/chat/share/...）
    app.include_router(chat.public_router, prefix=api_prefix)

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "app": settings.app_name}

    # 前端流式演示页：访问 http://127.0.0.1:8000/demo
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount(
            "/demo",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )

        @app.middleware("http")
        async def _no_cache_frontend(request, call_next):
            """静态资源加 no-cache：每次请求校验 ETag（有变化才重新下载）。

            解决"改了 index.html 但浏览器仍用旧版"的问题——StaticFiles 自带
            ETag，配合 no-cache 让浏览器每次都协商，命中 304 则用本地缓存。
            """
            if request.url.path.startswith("/demo"):
                resp = await call_next(request)
                resp.headers.setdefault("Cache-Control", "no-cache")
                return resp
            return await call_next(request)

    return app


# 供 uvicorn --factory 使用
app = create_app()
