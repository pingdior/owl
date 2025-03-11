@echo off
echo 检查Docker环境...

REM 检查Docker是否安装
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker未安装
    echo 在Windows上安装Docker的方法:
    echo 1. 访问 https://docs.docker.com/desktop/install/windows-install/ 下载Docker Desktop
    echo 2. 安装并启动Docker Desktop
    pause
    exit /b 1
)

echo Docker已安装

REM 检查Docker Compose是否安装
where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 警告: Docker Compose未找到，尝试使用新的docker compose命令
    docker compose version >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo 错误: Docker Compose未安装
        echo Docker Desktop for Windows应该已包含Docker Compose
        echo 请确保Docker Desktop已正确安装
        pause
        exit /b 1
    ) else (
        echo 使用新的docker compose命令
    )
) else (
    echo Docker Compose已安装
)

REM 检查Docker是否正在运行
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker未运行
    echo 请启动Docker Desktop应用程序
    pause
    exit /b 1
)

echo Docker正在运行

REM 检查是否有.env文件
if not exist "owl\.env" (
    echo 警告: 未找到owl\.env文件
    echo 请运行以下命令创建环境变量文件:
    echo copy owl\.env_template owl\.env
    echo 然后编辑owl\.env文件，填写必要的API密钥
) else (
    echo 环境变量文件已存在
)

echo 所有检查完成，您的系统已准备好构建和运行OWL项目的Docker容器
echo 请运行以下命令构建Docker镜像:
echo build_docker.bat

pause 