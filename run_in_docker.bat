@echo off
setlocal enabledelayedexpansion

REM 检查参数
if "%~1"=="" (
    echo 用法: run_in_docker.bat [脚本名称] "你的问题"
    echo 例如: run_in_docker.bat run.py "什么是人工智能？"
    echo 或者: run_in_docker.bat run_deepseek_example.py "什么是人工智能？"
    echo 如果不指定脚本名称，默认使用 run.py
    exit /b 1
)

REM 判断第一个参数是否是脚本名称
set SCRIPT_NAME=%~1
set QUERY=%~2

if "!SCRIPT_NAME:~-3!"==".py" (
    REM 如果提供了第二个参数，则为查询内容
    if "!QUERY!"=="" (
        echo 请提供查询参数，例如: run_in_docker.bat !SCRIPT_NAME! "你的问题"
        exit /b 1
    )
) else (
    REM 如果第一个参数不是脚本名称，则默认使用 run.py
    set QUERY=!SCRIPT_NAME!
    set SCRIPT_NAME=run.py
)

REM 检查脚本是否存在
if not exist "owl\!SCRIPT_NAME!" (
    echo 错误: 脚本 'owl\!SCRIPT_NAME!' 不存在
    echo 可用的脚本有:
    dir /b owl\*.py | findstr /v "__"
    exit /b 1
)

echo 使用脚本: !SCRIPT_NAME!
echo 查询内容: !QUERY!

REM 确保Docker容器正在运行
docker-compose ps | findstr "owl.*Up" > nul
if errorlevel 1 (
    echo 启动Docker容器...
    docker-compose up -d
    
    REM 等待容器启动
    echo 等待容器启动...
    timeout /t 5 /nobreak > nul
)

REM 在容器中运行指定的脚本，传递查询参数
echo 在Docker容器中运行脚本...
docker-compose exec -T owl xvfb-python !SCRIPT_NAME! "!QUERY!"

if errorlevel 0 (
    echo 查询完成！
) else (
    echo 查询执行失败，请检查错误信息。
)

pause 