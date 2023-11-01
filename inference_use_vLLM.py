import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from count import count_everything
import time
from vllm import LLM,SamplingParams


model_path = "/root/llm/Baichuan2-13B-Chat"

llm = LLM(model=model_path,trust_remote_code=True,dtype="half",tensor_parallel_size=4)
counter = count_everything()

while True:
    prompts = input("prompt:")
    sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=5000, 
            frequency_penalty=1.2
            )
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts=prompts,sampling_params=sampling_params)
    last_time = time.perf_counter()- start_time  # 时间单位s
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # token_ids = output.outputs[0].token_ids
        print("AI"+generated_text)

    speed = counter.count_token(generated_text,last_time)
    print(speed)
