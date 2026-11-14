@echo off
rem One-click start of the ANVESH officer screen (demonstration mode).
rem Starts the server on THIS computer only (127.0.0.1), signs in automatically and opens the browser.
rem Close this window (or press Ctrl+C) to stop ANVESH.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo.
echo   Starting ANVESH ... the browser opens by itself in a few seconds.
echo   If it does not, open  http://127.0.0.1:8760/
echo.
"%PY%" scripts\run_demo.py --reset --one-click --open
pause
