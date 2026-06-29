#!/bin/bash
cd "$(dirname "$0")"

# Find python in local or parent virtual environments
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXEC="venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_EXEC="../.venv/bin/python"
elif [ -f "../venv/bin/python" ]; then
    PYTHON_EXEC="../venv/bin/python"
elif [ -f "../../.venv/bin/python" ]; then
    PYTHON_EXEC="../../.venv/bin/python"
elif [ -f "../../venv/bin/python" ]; then
    PYTHON_EXEC="../../venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

$PYTHON_EXEC launcher_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "=================================================="
    echo "❌ 啟動失敗！(Launch Failed)"
    echo "請確認已安裝 Python 並且安裝了 tkinter 模組。"
    echo "=================================================="
    read -p "按任意鍵結束..." -n1 -s
fi
