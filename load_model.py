from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig
import streamlit as st
import torch


@st.cache_resource #缓存model，多次用相同参数调用function时，直接从cache返回结果，不重复执行function    
def load_model():

    ckpt_path = "/root/llm/baichuan-13B/Baichuan-13B-Chat"
    # from_pretrained()函数中，device_map参数的意思是把模型权重按指定策略load到多张GPU中
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,trust_remote_code=True,device_map="auto",torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,trust_remote_code=True)

    return model,tokenizer