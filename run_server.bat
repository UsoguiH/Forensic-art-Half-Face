@echo off
rem FaceLab demo server — watchdog: restarts uvicorn if it ever exits.
rem Runs independently of any Claude/terminal session. Log: server.log next
rem to this file (fresh file per restart so it cannot grow unbounded).
set FACELAB_PROVIDER=fal
cd /d "%~dp0backend"
:loop
python -X utf8 -m uvicorn backend:app --host 127.0.0.1 --port 8000 > "%~dp0server.log" 2>&1
timeout /t 3 /nobreak >nul
goto loop
