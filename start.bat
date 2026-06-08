@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   猫狗识别 - 启动本地服务
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/2] 检查依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo [2/2] 启动服务并打开浏览器...
echo.
python server.py

pause
