# 开发与验证

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,local-embedding]"
```

`pyproject.toml` 是依赖版本的唯一来源；`requirements.txt` 仅作为兼容入口。

## 启动

复制 `.env.example` 为 `.env`，填写模型供应商配置，然后运行：

```powershell
python run.py
```

健康检查为 `GET /health`，演示页为 `/demo`。

## 测试

```powershell
pytest
pytest -m network
pytest -m chroma
```

默认套件使用 `hashing://512` 确定性嵌入，不下载模型，也不要求 Redis、Ollama 或 Chroma。真实外部服务测试通过 marker 显式执行。

测试运行数据统一写入被忽略的 `var/`。当前受限 Windows 环境会使用项目级 `tmp_path` fixture，避免系统临时目录权限导致误报。

## 主要边界

- API 路由：参数、鉴权、响应适配。
- `app/services`：会话、分享和管理员业务。
- `app/agent`：编排、事件、上下文、收尾。
- `app/rag`：统一入库与检索管线。
- `app/knowledge/stores`：向量存储后端。
- `app/memory`：短期与长期记忆。

更完整的依赖约定参见 `docs/architecture.md`。
