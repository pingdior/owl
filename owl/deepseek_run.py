from dotenv import load_dotenv
load_dotenv()

from camel.models import ModelFactory
from camel.toolkits import *
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from typing import List, Dict

from retry import retry
from loguru import logger

from utils import OwlRolePlaying, run_society
import os
import time
from playwright.sync_api import sync_playwright
import inspect

# 在文件顶部添加
print(f"使用API平台: DEEPSEEK")
print(f"API密钥已设置: {'DEEPSEEK_API_KEY' in os.environ}")
print(f"搜索引擎ID已设置: {bool(os.environ.get('SEARCH_ENGINE_ID', ''))}")

# 添加在构建工具列表之前
print("WebToolkit支持的参数:")
print(inspect.signature(WebToolkit.__init__))

def get_webpage_content(url):
    """使用Playwright获取网页内容"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=60000)
            content = page.content()
            text = page.evaluate("""() => {
                return document.body.innerText;
            }""")
            browser.close()
            return text
        except Exception as e:
            browser.close()
            return f"获取网页内容失败: {str(e)}"

def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    # 创建DeepSeek兼容的配置
    deepseek_config = {
        "temperature": 0.7,       # 控制随机性
        "top_p": 1.0,             # 控制输出多样性
        "max_tokens": 4096,       # 最大输出令牌数
        "presence_penalty": 0.0,  # 控制主题重复性
        "frequency_penalty": 0.0, # 控制词汇重复性
        # DeepSeek可能支持的其他参数
    }
    
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
        model_config_dict=deepseek_config,  # 使用兼容配置
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
        model_config_dict=deepseek_config,  # 使用兼容配置
    )

    # 使用最基本的WebToolkit配置
    tools_list = [
        *WebToolkit(
            headless=False,  # 这是一个肯定支持的参数
            web_agent_model=assistant_model,
            planning_agent_model=assistant_model
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *SearchToolkit(model=assistant_model).get_tools(),
        *ExcelToolkit().get_tools()
    ]

    user_role_name = 'user'
    user_agent_kwargs = dict(model=user_model)
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = dict(model=assistant_model,
    tools=tools_list)
    
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


# Example case
question = "使用Python计算1到100的和并解释代码"

# 或使用Google搜索相关问题
question = "简要介绍什么是元宇宙？"

if question.startswith("http"):
    print("检测到URL，直接获取内容...")
    url = question.split()[0]
    content = get_webpage_content(url)
    question = f"请总结以下内容：\n{content[:5000]}..."  # 限制内容长度

time.sleep(3)  # 增加间隔时间，避免API限制
society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")





