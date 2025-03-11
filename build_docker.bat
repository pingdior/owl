@echo off
echo 在Windows上构建Docker镜像...

REM 设置Docker BuildKit环境变量
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

echo 启用Docker BuildKit加速构建...

REM 创建缓存目录
if not exist ".docker-cache\pip" mkdir .docker-cache\pip

REM 构建Docker镜像
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1

if %ERRORLEVEL% EQU 0 (
    echo Docker镜像构建成功！
    echo 可以使用以下命令启动容器：
    echo docker-compose up -d
) else (
    echo Docker镜像构建失败，请检查错误信息。
)

pause 