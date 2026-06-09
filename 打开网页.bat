@echo off
chcp 65001 >nul
echo 正在打开 http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"
echo.
echo 如果页面打不开，请先双击 start.bat 启动服务
pause
