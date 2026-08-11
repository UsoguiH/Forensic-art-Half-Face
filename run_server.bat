@echo off
rem FaceLab demo server — watchdog: restarts uvicorn if it ever exits.
rem Runs independently of any Claude/terminal session. Log: server.log next
rem to this file (fresh file per restart so it cannot grow unbounded).
set FACELAB_PROVIDER=fal
rem 2026-08-11 user decision: FLUX.2 dev is the engine (matches offline lab).
rem Benchmarked trade-off vs qwen is logged in Test_AI_Half/NewDataSet_Bench.
rem Brush/region edits also lead with FLUX; the verified retry chain still
rem falls back to qwen only if FLUX fails to apply the edit or is refused.
set FAL_EDIT_MODEL=dev
set FACELAB_REGION_MODEL=dev
cd /d "%~dp0backend"
:loop
python -X utf8 -m uvicorn backend:app --host 127.0.0.1 --port 8000 > "%~dp0server.log" 2>&1
timeout /t 3 /nobreak >nul
goto loop
