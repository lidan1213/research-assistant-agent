"""Root conftest：确保本项目自己的 `app` 包始终优先被解析。

修复一个隐蔽坑：当本机存在另一个「同名 `app` 包」的 editable 安装
（例如另一项目的 backend/app 被 `pip install -e` 进了同一个解释器）时，
若从子目录或共享解释器启动测试，`import app` 可能错误地加载到别的项目，
从而触发配置校验失败（extra_forbidden 等）。

把项目根目录插到 sys.path 最前，可保证无论以何种方式启动，`app` 都指向本项目。
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep pytest temporary files inside the writable workspace on restricted Windows hosts.
PYTEST_TEMP_ROOT = os.path.join(ROOT, "var", "pytest-system-temp")
os.makedirs(PYTEST_TEMP_ROOT, exist_ok=True)
os.environ["TMP"] = PYTEST_TEMP_ROOT
os.environ["TEMP"] = PYTEST_TEMP_ROOT


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local replacement for hosts that restrict pytest's temp owner dirs."""
    path = Path(ROOT) / "var" / "test-runs" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
