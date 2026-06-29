@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=.venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=..\.venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=..\.venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=..\venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=..\venv\Scripts\python.exe"
) else if exist "..\..\.venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=..\..\.venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=..\..\.venv\Scripts\python.exe"
) else if exist "..\..\venv\Scripts\pythonw.exe" (
    set "PYTHON_EXEC=..\..\venv\Scripts\pythonw.exe"
    set "PYTHON_CLI=..\..\venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=pythonw"
    set "PYTHON_CLI=python"
)

:: Try running a quick import check with the CLI Python.
:: If tkinter or anything crashes, we show the console error.
"%PYTHON_CLI%" -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo ==================================================
    echo ❌ 啟動失敗！(Launch Failed)
    echo 找不到 tkinter 模組。請確認您的 Python 環境已正確安裝。
    echo ==================================================
    pause
    exit /b %errorlevel%
)

start "" "%PYTHON_EXEC%" launcher_gui.py
