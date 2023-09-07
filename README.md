# langchain-llm
Build large model applications through Langchain

本项目基于baichuan-13b基座大模型，使用langchain框架探索AI Agent,Tools的应用。
1. 用fastapi实现baichuan的api
2. 使用langchain的LLM wrapper包装baichuan api，使其成为langchain的一个LLM对象
3. 应用1: 使用langchain获取arxive论文，并总结摘要
