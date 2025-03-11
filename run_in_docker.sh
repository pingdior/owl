#!/bin/bash

# 检测操作系统类型
OS_TYPE=$(uname -s)
echo "检测到操作系统: $OS_TYPE"

# 检查是否提供了查询参数
if [ $# -lt 1 ]; then
    echo "用法: ./run_in_docker.sh [脚本名称] '你的问题'"
    echo "例如: ./run_in_docker.sh run.py '什么是人工智能？'"
    echo "或者: ./run_in_docker.sh run_deepseek_example.py '什么是人工智能？'"
    echo "如果不指定脚本名称，默认使用 run.py"
    exit 1
fi

# 判断第一个参数是否是脚本名称
if [[ $1 == *.py ]]; then
    SCRIPT_NAME="$1"
    # 如果提供了第二个参数，则为查询内容
    if [ $# -ge 2 ]; then
        QUERY="$2"
    else
        echo "请提供查询参数，例如: ./run_in_docker.sh $SCRIPT_NAME '你的问题'"
        exit 1
    fi
else
    # 如果第一个参数不是脚本名称，则默认使用 run.py
    SCRIPT_NAME="run.py"
    QUERY="$1"
fi

# 根据操作系统类型设置脚本路径检查方式
if [[ "$OS_TYPE" == "Darwin" ]] || [[ "$OS_TYPE" == "Linux" ]]; then
    # macOS 或 Linux
    if [ ! -f "owl/$SCRIPT_NAME" ]; then
        echo "错误: 脚本 'owl/$SCRIPT_NAME' 不存在"
        echo "可用的脚本有:"
        ls -1 owl/*.py | grep -v "__"
        exit 1
    fi
else
    # Windows
    if [ ! -f "owl/$SCRIPT_NAME" ]; then
        echo "错误: 脚本 'owl/$SCRIPT_NAME' 不存在"
        echo "可用的脚本有:"
        find owl -name "*.py" | grep -v "__" | sed 's/\\/\//g'
        exit 1
    fi
fi

echo "使用脚本: $SCRIPT_NAME"
echo "查询内容: $QUERY"

# 确保Docker容器正在运行
CONTAINER_RUNNING=$(docker-compose ps | grep -c "owl.*Up" || true)
if [ "$CONTAINER_RUNNING" -eq 0 ]; then
    echo "启动Docker容器..."
    docker-compose up -d
    
    # 等待容器启动
    echo "等待容器启动..."
    sleep 5
fi

# 直接在容器中运行指定的脚本，传递查询参数
echo "在Docker容器中运行脚本..."

# 根据操作系统类型执行不同的命令
if [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
    # Windows可能需要特殊处理引号
    winpty docker-compose exec -T owl xvfb-python $SCRIPT_NAME "$QUERY"
else
    # macOS 或 Linux
    docker-compose exec -T owl xvfb-python $SCRIPT_NAME "$QUERY"
fi

# 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "查询完成！"
else
    echo "查询执行失败，请检查错误信息。"
fi 