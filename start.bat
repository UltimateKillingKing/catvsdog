@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   猫狗识别 - 启动本地服务
echo ========================================
echo.
echo 当前目录: %cd%
echo.

REM 优先使用 py 启动器，避免 Windows 应用商店 Python 假入口
set PYTHON_CMD=
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 --version >nul 2>&1
    if %errorlevel%==0 set PYTHON_CMD=py -3
)
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 (
        python --version >nul 2>&1
        if %errorlevel%==0 set PYTHON_CMD=python
    )
)
if not defined PYTHON_CMD (
    echo [错误] 未检测到可用的 Python，请安装 Python 3.8+ 并勾选 Add to PATH
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 使用 Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

if not exist "server.py" (
    echo [错误] 找不到 server.py，请确认在正确目录运行此脚本
    echo 正确路径应包含: server.py, index.html, start.bat
    pause
    exit /b 1
)

if not exist "best_resnet50_cat_dog.pth" (
    echo [错误] 找不到模型文件 best_resnet50_cat_dog.pth
    pause
    exit /b 1
)

REM 如果服务已在运行，直接打开浏览器
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo [提示] 服务已在运行，正在打开浏览器...
    start "" "http://127.0.0.1:5000"
    echo.
    echo 浏览器地址: http://127.0.0.1:5000
    pause
    exit /b 0
)

echo [1/2] 检查依赖（首次可能较慢）...
%PYTHON_CMD% -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试
    pause
    exit /b 1
)

echo [2/2] 启动服务...
echo.
echo 启动成功后浏览器会自动打开
echo 若未自动打开，请手动访问: http://127.0.0.1:5000
echo 关闭此窗口将停止服务
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"
%PYTHON_CMD% server.py

pause
