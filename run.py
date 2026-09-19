"""项目启动入口：先确保本项目 `app` 优先解析，再启动 uvicorn。

用法（在项目根目录下）：
    python run.py                 # 开发模式（reload 热重载）
    python run.py --port 8080    # 指定端口
    python run.py --no-reload    # 关闭热重载（生产/调试用）

等价命令：python -m uvicorn app.main:app --reload
"""
import os
import sys

# 把项目根目录插到 sys.path 最前，避免机器上其它同名 `app` 包被误加载
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ResearchAssistantAgent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=True)
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
