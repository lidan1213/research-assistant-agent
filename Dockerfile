# 科研助手 Agent — 生产镜像
# 基于 Python 3.11 slim；前端由后端 /demo 静态托管，单容器即可。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY .env ./.env

EXPOSE 8000

CMD ["python", "scripts/run.py"]
