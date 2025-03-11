#!/bin/bash

# 检测操作系统类型
OS_TYPE=$(uname -s)
echo "检测到操作系统: $OS_TYPE"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "在macOS上安装Docker的方法:"
        echo "1. 访问 https://docs.docker.com/desktop/install/mac-install/ 下载Docker Desktop"
        echo "2. 安装并启动Docker Desktop"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        echo "在Linux上安装Docker的方法:"
        echo "1. 运行以下命令:"
        echo "   sudo apt-get update"
        echo "   sudo apt-get install docker.io docker-compose"
        echo "2. 启动Docker服务:"
        echo "   sudo systemctl start docker"
        echo "   sudo systemctl enable docker"
    elif [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
        echo "在Windows上安装Docker的方法:"
        echo "1. 访问 https://docs.docker.com/desktop/install/windows-install/ 下载Docker Desktop"
        echo "2. 安装并启动Docker Desktop"
    fi
    
    exit 1
fi

echo "Docker已安装"

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装"
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "Docker Desktop for Mac已包含Docker Compose"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        echo "在Linux上安装Docker Compose的方法:"
        echo "1. 运行以下命令:"
        echo "   sudo apt-get install docker-compose"
    elif [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
        echo "Docker Desktop for Windows已包含Docker Compose"
    fi
    
    exit 1
fi

echo "Docker Compose已安装"

# 检查Docker是否正在运行
if ! docker info &> /dev/null; then
    echo "错误: Docker未运行"
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "请启动Docker Desktop应用程序"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        echo "请运行以下命令启动Docker服务:"
        echo "sudo systemctl start docker"
    elif [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
        echo "请启动Docker Desktop应用程序"
    fi
    
    exit 1
fi

echo "Docker正在运行"

# 检查是否有足够的磁盘空间
FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "可用磁盘空间: $FREE_SPACE"

# 检查是否有.env文件
if [ ! -f "owl/.env" ]; then
    echo "警告: 未找到owl/.env文件"
    echo "请运行以下命令创建环境变量文件:"
    echo "cp owl/.env_template owl/.env"
    echo "然后编辑owl/.env文件，填写必要的API密钥"
else
    echo "环境变量文件已存在"
fi

echo "所有检查完成，您的系统已准备好构建和运行OWL项目的Docker容器"
echo "请运行以下命令构建Docker镜像:"

if [[ "$OS_TYPE" == MINGW* ]] || [[ "$OS_TYPE" == CYGWIN* ]] || [[ "$OS_TYPE" == MSYS* ]]; then
    echo "build_docker.bat"
else
    echo "./build_docker.sh"
fi 