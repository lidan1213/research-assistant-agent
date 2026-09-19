@echo off
REM ============================================================
REM  Research Assistant Agent - One-click launcher
REM  Starts: ChromaDB (0.4.24, port 8001) + Main service (8000)
REM  IMPORTANT: ChromaDB MUST be 0.4.x (.chroma-venv).
REM  The .venv copy is 1.5.9 whose v1 API is deprecated and
REM  will MIGRATE (and break) the 0.4.x data directory.
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] Checking ChromaDB version (.chroma-venv) ...
.chroma-venv\Scripts\python.exe -c "import chromadb;v=chromadb.__version__;assert v.startswith('0.4'), 'BAD VERSION '+v;print('  ChromaDB '+v+' OK')" >nul 2>&1
if errorlevel 1 (
  echo   [ERROR] .chroma-venv does not have ChromaDB 0.4.x.
  echo   Fix: PYTHONPATH= .chroma-venv\Scripts\pip.exe install "chromadb==0.4.24"
  echo   Aborting to protect data/knowledge_chroma from 1.x migration.
  pause
  exit /b 1
)
echo   ChromaDB 0.4.24 OK

echo [2/4] Starting ChromaDB on :8001 ...
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if errorlevel 1 (
  start "ChromaDB-8001" cmd /k "set ANONYMIZED_TELEMETRY=False&& .chroma-venv\Scripts\chroma.exe run --path data\knowledge_chroma --port 8001 --host 127.0.0.1"
  echo   ChromaDB launching in new window...
) else (
  echo   ChromaDB already running on :8001
)

echo [3/4] Starting main service on :8000 ...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 (
  start "ResearchAgent-8000" cmd /k "set APP_ENV=production&& set PYTHONPATH=&& .venv\Scripts\python.exe scripts\run.py"
  echo   Main service launching in new window...
) else (
  echo   Main service already running on :8000
)

echo [4/4] Waiting for services to become ready ...
timeout /t 8 /nobreak >nul

REM --- verify ChromaDB (8001) ---
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if errorlevel 1 (
  echo   [WARN] ChromaDB not listening on :8001 yet.
  echo   If the window did not open, start it manually:
  echo     cd /d "%~dp0"
  echo     PYTHONPATH= .chroma-venv\Scripts\chroma.exe run --path data\knowledge_chroma --port 8001 --host 127.0.0.1
) else (
  echo   ChromaDB :8001 OK
)

REM --- verify main service (8000) ---
curl -s -o nul -w "  Main service :8000 -> HTTP %%{http_code}" http://127.0.0.1:8000/health
echo.
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 (
  echo   [WARN] Main service not listening on :8000.
  echo   Manual: set APP_ENV=production&& set PYTHONPATH=&& .venv\Scripts\python.exe scripts\run.py
)

echo.
echo ============================================================
echo  Frontend : http://127.0.0.1:8000/demo
echo  API      : http://127.0.0.1:8000
echo  ChromaDB : http://127.0.0.1:8001
echo ============================================================
pause
