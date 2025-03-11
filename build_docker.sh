#!/bin/bash

# 检测操作系统类型
OS_TYPE=$(uname -s)
echo "检测到操作系统: $OS_TYPE"

# 设置Docker BuildKit环境变量
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "启用Docker BuildKit加速构建..."

# 创建缓存目录
mkdir -p .docker-cache/pip

# 根据操作系统类型执行不同的命令
if [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS
    echo "在macOS上构建Docker镜像..."
    docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
elif [[ "$OS_TYPE" == "Linux" ]]; then
    # Linux
    echo "在Linux上构建Docker镜像..."
    docker-compose build --parallel --build-arg BUILDKIT_INLINE_CACHE=1
elif [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
    # Windows
    echo "在Windows上构建Docker镜像..."
    docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
else
    echo "未知操作系统，尝试使用标准命令构建..."
    docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
fi

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "Docker镜像构建成功！"
    echo "可以使用以下命令启动容器："
    echo "docker-compose up -d"
else
    echo "Docker镜像构建失败，请检查错误信息。"
fi 