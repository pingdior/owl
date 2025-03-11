from dotenv import load_dotenv
load_dotenv()
from camel.models import ModelFactory
from camel.toolkits import (
    WebToolkit, 
    DocumentProcessingToolkit, 
    VideoAnalysisToolkit, 
    AudioAnalysisToolkit, 
    CodeExecutionToolkit, 
    ImageAnalysisToolkit, 
    SearchToolkit,
    ExcelToolkit,
    FunctionTool
    )
from camel.types import ModelPlatformType
from camel.configs import QwenConfig
from loguru import logger
from utils import OwlRolePlaying, run_society
import os

# 从环境变量获取API密钥，而不是硬编码
qwen_api_key = os.environ.get('QWEN_API_KEY')

# 检查API密钥是否存在
if not qwen_api_key:
    raise ValueError("QWEN_API_KEY环境变量未设置。请在.env文件中设置此变量。")

def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""
    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-max",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
    assistant_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-max",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
    search_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-max",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
    planning_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-max",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
    web_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-vl-plus-latest",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
    multimodal_model = ModelFactory.create(
             model_platform=ModelPlatformType.QWEN,
             model_type="qwen-vl-plus-latest",
             model_config_dict={"temperature": 0.3},
             api_key=qwen_api_key,
         )
         
    tools_list = [
        *WebToolkit(
            headless=False, 
            web_agent_model=web_model, 
            planning_agent_model=planning_model
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *VideoAnalysisToolkit(model=multimodal_model).get_tools(),  # This requires multimodal model
        *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit().get_tools(),
        *ImageAnalysisToolkit(model=multimodal_model).get_tools(),  # This requires multimodal model
        *SearchToolkit(model=search_model).get_tools(),
        #FunctionTool(SearchToolkit(model=search_model).search_duckduckgo),如果没有Google相关api可以用duckduckgo代替
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
        'output_language': '中文',
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
question = "在百度热搜上，查看第一条新闻，然后给我一个总结报告"
society = construct_society(question)
answer, chat_history, token_count = run_society(society)
logger.success(f"Answer: {answer}")