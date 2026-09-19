"""服务守护进程：探测主服务(:8000) 与 ChromaDB(:8001)，失败自动重启。

背景：主服务与 ChromaDB 曾静默崩溃（无通知），无人值守时服务不可用。
本脚本每 --interval 秒探测一次 /health 与 heartbeat；连续失败
--fail-threshold 次后自动拉起对应进程（restart 防抖），日志写入 data/watchdog.log。

用法（独立窗口运行，保持常驻）：
  PYTHONPATH= .venv/Scripts/python.exe scripts/watchdog.py
  PYTHONPATH= .venv/Scripts/python.exe scripts/watchdog.py --interval 20 --fail-threshold 2
  PYTHONPATH= .venv/Scripts/python.exe scripts/watchdog.py --once      # 单次探测（诊断用）
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "watchdog.log"

WEB_URL = "http://127.0.0.1:8000/health"
CHROMA_URL = "http://127.0.0.1:8001/api/v1/heartbeat"

# 重启命令（detached，独立进程组，不受本脚本退出影响）
WEB_CMD = [
    str(ROOT / ".venv" / "Scripts" / "python.exe"), "scripts/run.py",
]
CHROMA_CMD = [
    str(ROOT / ".chroma-venv" / "Scripts" / "chroma.exe"),
    "run", "--path", "data/knowledge_chroma", "--port", "8001", "--host", "127.0.0.1",
]


def _log(msg: str) -> None:
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def _port_busy(port: int) -> bool:
    """端口是否已有 LISTEN（避免重启时与残留进程冲突 bind 10048）。"""
    try:
        # Windows 中文系统 netstat 输出 GBK；errors=replace 兼容任何代码页
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            encoding="gbk", errors="replace", timeout=10,
        ).stdout
        return any(f":{port}" in line and "LISTENING" in line for line in out.splitlines())
    except Exception:  # noqa: BLE001
        return False  # netstat 失败时保守认为空闲


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn(cmd: list[str], name: str, env_extra: dict[str, str]) -> None:
    """detached 拉起进程（独立控制台，日志重定向到 data/logs/）。"""
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = ""  # 防止 Hermes/外部 venv 污染
    with open(log_dir / f"{name}.log", "a", encoding="utf-8") as out:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0),
        )


def _ensure(name: str, url: str, port: int, cmd: list[str], env_extra: dict[str, str]) -> None:
    if _probe(url):
        return  # 健康
    if _port_busy(port):
        _log(f"[WARN] {name} 探测失败但端口 {port} 被占用——疑似半死进程，跳过自动重启，请人工检查")
        return
    _log(f"[RESTART] {name} 探测失败（{url}），自动拉起…")
    try:
        _spawn(cmd, name, env_extra)
        time.sleep(5)
        if _probe(url):
            _log(f"[OK] {name} 重启成功")
        else:
            _log(f"[WARN] {name} 已拉起但 5s 后仍未就绪（等待启动）")
    except Exception as e:  # noqa: BLE001
        _log(f"[ERROR] {name} 重启失败: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="科研助手服务守护（8000 + 8001）")
    parser.add_argument("--interval", type=int, default=30, help="探测间隔秒（默认 30）")
    parser.add_argument("--fail-threshold", type=int, default=3, help="连续失败次数后重启（默认 3）")
    parser.add_argument("--once", action="store_true", help="单次探测后退出（诊断用）")
    args = parser.parse_args()

    _log(f"watchdog 启动：interval={args.interval}s fail_threshold={args.fail_threshold} "
         f"web={WEB_URL} chroma={CHROMA_URL}")
    fails_web = fails_chroma = 0
    while True:
        web_ok = _probe(WEB_URL)
        chroma_ok = _probe(CHROMA_URL)
        fails_web = fails_web + 1 if not web_ok else 0
        fails_chroma = fails_chroma + 1 if not chroma_ok else 0
        if fails_web >= args.fail_threshold:
            _ensure("web", WEB_URL, 8000, WEB_CMD, {"APP_ENV": "production"})
            fails_web = 0
        if fails_chroma >= args.fail_threshold:
            _ensure("chroma", CHROMA_URL, 8001, CHROMA_CMD,
                    {"ANONYMIZED_TELEMETRY": "False"})
            fails_chroma = 0
        if args.once:
            _log(f"web={'OK' if web_ok else 'DOWN'} | chroma={'OK' if chroma_ok else 'DOWN'}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
