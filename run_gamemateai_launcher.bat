@echo off
setlocal

REM MSI Gaming AI Assistant launcher helper
set "BASE=%~dp0"

REM Prefer bundled venv pythonw, then python
if exist "%BASE%\.venv\Scripts\pythonw.exe" (
    set "PYTHON=%BASE%\.venv\Scripts\pythonw.exe"
) else if exist "%BASE%\.venv\Scripts\python.exe" (
    set "PYTHON=%BASE%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=pythonw"
)

cd /d "%BASE%"
"%PYTHON%" gamemateai_launcher.py

endlocal
