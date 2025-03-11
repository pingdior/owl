<h1 align="center">
	🦉 OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
  🦉 OWL: 优化劳动力学习的通用智能体，用于处理现实世界的自动化任务
</h1>


<div align="center">

[![文档][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![微信][wechat-image]][wechat-url]
[![微信][owl-image]][owl-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![软件许可证][package-license-image]][package-license-url]


</div>


<hr>

<div align="center">
<h4 align="center">

[English README](https://github.com/camel-ai/owl/tree/main) |
[社区](https://github.com/camel-ai/camel#community) |
[安装](#️-installation) |
[示例](https://github.com/camel-ai/owl/tree/main/owl) |
[论文](https://arxiv.org/abs/2303.17760) |
[引用](#-community) |
[贡献](https://github.com/camel-ai/owl/graphs/contributors) |
[CAMEL-AI](https://www.camel-ai.org/)

</h4>

<div align="center" style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin: 15px 0;">
  <h3 style="color: #1e88e5; margin: 0;">
    🏆 OWL 在 GAIA 基准测试中取得 <span style="color: #d81b60; font-weight: bold; font-size: 1.2em;">58.18</span> 平均分，在开源框架中排名 <span style="color: #d81b60; font-weight: bold; font-size: 1.2em;">🏅️ #1</span>！ 🏆
  </h3>
</div>

<div align="center">

🦉 OWL 是一个前沿的多智能体协作框架，推动任务自动化的边界，构建在 [CAMEL-AI Framework](https://github.com/camel-ai/camel)。

我们的愿景是彻底变革 AI 智能体协作解决现实任务的方式。通过利用动态智能体交互，OWL 实现了跨多领域更自然、高效且稳健的任务自动化。

</div>

![](./assets/owl_architecture.png)



<br>


</div>

<!-- # Key Features -->
# 📋 目录

- [📋 目录](#-目录)
- [🔥 新闻](#-新闻)
- [🎬 演示视频](#-演示视频)
- [✨️ 核心功能](#-核心功能)
- [🛠️ 安装](#️-安装)
  - [**克隆 Github 仓库**](#克隆-github-仓库)
  - [**设置环境**](#设置环境)
  - [**安装依赖**](#安装依赖)
  - [**设置环境变量**](#设置环境变量)
- [🚀 快速开始](#-快速开始)
- [🧪 实验](#-实验)
- [⏱️ 未来计划](#️-未来计划)
- [📄 许可证](#-许可证)
- [🖊️ 引用](#️-引用)
- [🔥 社区](#-社区)
- [❓ 常见问题](#-常见问题)


# 🔥 新闻

- **[2025.03.07]**: 我们开源了 🦉 OWL 项目的代码库。

# 🎬 演示视频

https://private-user-images.githubusercontent.com/55657767/420211368-f29f477d-7eef-46da-8d7a-8f3bcf506da2.mp4

https://private-user-images.githubusercontent.com/55657767/420212194-e813fc05-136a-485f-8df3-f10d9b4e63ec.mp4

# ✨️ 核心功能

- **在线搜索**：使用维基百科、谷歌搜索等，进行实时信息检索
- **多模态处理**：支持互联网或本地视频、图片、语音处理
- **浏览器操作**：借助Playwright框架开发浏览器模拟交互，支持页面滚动、点击、输入、下载、历史回退等功能
- **文件解析**：word、excel、PDF、PowerPoint信息提取，内容转文本/Markdown
- **代码执行**：编写python代码，并使用解释器运行

# 🛠️ 安装

## **克隆 Github 仓库**

```bash
git clone https://github.com/camel-ai/owl.git
cd owl
```

## **设置环境**

使用 Conda（推荐）：
```bash
conda create -n owl python=3.11
conda activate owl
```

使用 venv（备用）：
```bash
python -m venv owl_env
# Windows 系统
owl_env\Scripts\activate
# Unix 或 MacOS 系统
source owl_env/bin/activate
```

## **安装依赖**

```bash
python -m pip install -r requirements.txt
```

## **设置环境变量**  

在 `owl/.env_template` 文件中，你可以找到所有必要的 API 密钥以及各服务的注册网址。要使用这些 API 服务，请按照以下步骤操作：

1. *复制并重命名*: 复制 `.env_example` 文件，并将副本重命名为 `.env`。
2. *填写你的密钥*: 打开 `.env` 文件，在相应字段中填入你的 API 密钥。 
3. *如需使用更多其他模型*：请参考我们CAMEL的models文档：https://docs.camel-ai.org/key_modules/models.html#supported-model-platforms-in-camel

> **注意**：为获得最佳性能，我们强烈建议使用 OpenAI 模型。我们通过测试发现，其他模型在处理复杂任务和基准测试时可能会导致性能显著降低。

# 🚀 快速开始
   
运行以下最小示例：

```bash
python owl/run.py
```

你可以通过修改 `run.py` 来运行自定义任务的 OWL 智能体：

```python
# Define your own task
question = "Task description here."

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")
```

你可以尝试以下示例任务：
- "查询苹果公司的最新股票价格"
- "分析关于气候变化的最新推文情绪"
- "帮我调试这段 Python 代码：[在此粘贴你的代码]"
- "总结这篇研究论文的主要观点：[论文URL]"
# 🧪 实验

我们提供了一个脚本用于复现 GAIA 上的实验结果。  
你可以查看 `run_gaia_roleplaying.py` 文件，并运行以下命令：

```bash
python run_gaia_roleplaying.py
```

# ⏱️ 未来计划

- [ ] 撰写一篇技术博客，详细介绍我们在现实任务中多智能体协作方面的探索与见解。
- [ ] 通过引入更多针对特定领域任务的专业工具，进一步完善工具生态系统。
- [ ] 开发更复杂的智能体交互模式和通信协议


# 📄 许可证

源代码采用 Apache 2.0 许可证。

# 🖊️ 引用

如果你觉得这个仓库对你有帮助，请引用：


```
@misc{owl2025,
  title        = {OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation},
  author       = {{CAMEL-AI.org}},
  howpublished = {\url{https://github.com/camel-ai/owl}},
  note         = {Accessed: 2025-03-07},
  year         = {2025}
}
```

# 🔥 社区
加入我们，参与更多讨论！
<!-- ![](./assets/community.png) -->
![](./assets/community_4.jpg)
![](./assets/meetup.jpg)

# ❓ 常见问题

**Q: 为什么我的Chrome浏览器显示空白页面，但控制台有输出结果？**

A: 这是预期的行为。当OWL判断某个任务可以使用非浏览器工具（如搜索、代码分析等）完成时，浏览器窗口可能保持空白。浏览器仅在需要网页交互时才会被激活。我们计划在未来的更新中实现延迟加载以改善这一用户体验。


[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://camel-ai.github.io/camel/index.html
[star-image]: https://img.shields.io/github/stars/camel-ai/owl?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/owl/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/owl/blob/main/licenses/LICENSE

[colab-url]: https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[huggingface-url]: https://huggingface.co/camel-ai
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[discord-url]: https://discord.camel-ai.org/
[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb
[wechat-url]: https://ghli.org/camel/wechat.png
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[x-url]: https://x.com/CamelAIOrg
[x-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social
[twitter-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter
[reddit-url]: https://www.reddit.com/r/CamelAI/
[reddit-image]: https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white
[ambassador-url]: https://www.camel-ai.org/community
[owl-url]: ./assets/qr_code.jpg
[owl-image]: https://img.shields.io/badge/WeChat-OWLProject-brightgreen?logo=wechat&logoColor=white
