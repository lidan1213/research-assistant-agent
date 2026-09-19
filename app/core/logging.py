"""结构化日志配置（基于 loguru）。

统一日志格式，便于在云原生环境中被采集（JSON / 文本均可切换）。
控制台输出 + 文件输出（日切轮转，保留 14 天，重启不丢）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

from app.config import get_settings


def setup_logging(level: str | None = None, fmt: Literal["text", "json"] = "text") -> None:
    """初始化全局 logger：控制台 + 文件（日切轮转）。"""
    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    logger.remove()
    if fmt == "json":
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            colorize=settings.app_env == "development",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[component]}</cyan> | "
                "{message}"
            ),
        )

    # 文件输出：日切轮转（app-YYYY-MM-DD.log），保留 14 天；重启不丢历史
    log_dir = Path(settings.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "app-{time:YYYY-MM-DD}.log",
            level=log_level,
            rotation=["00:00", "100 MB"],   # 每天零点切分；单日超 100MB 也切
            retention="14 days",       # 保留 14 天
            encoding="utf-8",
            enqueue=True,              # 多线程/多进程安全（Windows 单进程也安全）
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{extra[component]} | {message}"
            ),
        )
        # error 单独输出，便于快速定位故障（与全量日志同轮转策略）
        logger.add(
            log_dir / "error-{time:YYYY-MM-DD}.log",
            level="ERROR",
            rotation=["00:00", "50 MB"],
            retention="30 days",
            encoding="utf-8",
            enqueue=True,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{extra[component]} | {message}"
            ),
        )
    except Exception:  # noqa: BLE001
        # 文件日志失败（如只读目录/沙箱）不应阻塞启动，仅退化为控制台
        pass


def get_logger(component: str = "app"):
    """获取带 component 标签的 logger。"""
    return logger.bind(component=component)
