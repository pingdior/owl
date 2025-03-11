FROM python:3.10-slim

WORKDIR /app

# 设置pip镜像源以加速下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 优化apt安装，减少层数
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    # 添加xvfb和相关依赖
    xvfb \
    xauth \
    x11-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 先复制并安装requirements.txt，利用Docker缓存机制
COPY requirements.txt .
# 启用pip缓存以加速构建
RUN pip install -r requirements.txt

# 安装 Playwright 依赖（使用国内镜像源）
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
ENV PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright
RUN pip install playwright && \
    playwright install --with-deps chromium

# 复制项目文件
COPY owl/ ./owl/
COPY licenses/ ./licenses/
COPY assets/ ./assets/
COPY README.md .
COPY README_zh.md .

# 设置环境变量文件
COPY owl/.env_template ./owl/.env

# 设置工作目录
WORKDIR /app/owl

# 创建启动脚本
RUN echo '#!/bin/bash\nxvfb-run --auto-servernum --server-args="-screen 0 1280x960x24" python "$@"' > /usr/local/bin/xvfb-python && \
    chmod +x /usr/local/bin/xvfb-python

# 创建欢迎脚本
RUN echo '#!/bin/bash\necho "欢迎使用OWL项目Docker环境！"\necho ""\necho "可用的脚本:"\nls -1 *.py | grep -v "__" | sed "s/^/- /"\necho ""\necho "运行示例:"\necho "  xvfb-python run.py                     # 运行默认脚本"\necho "  xvfb-python run_deepseek_example.py      # 运行DeepSeek示例"\necho ""\necho "或者使用自定义查询:"\necho "  xvfb-python run.py \"你的问题\""\necho ""' > /usr/local/bin/owl-welcome && \
    chmod +x /usr/local/bin/owl-welcome

# 容器启动命令（改为交互式shell）
CMD ["/bin/bash", "-c", "owl-welcome && /bin/bash"] 