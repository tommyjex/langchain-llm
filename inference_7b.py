from transformers import AutoModelForCausalLM, AutoTokenizer
from count import count_everything
import time
import torch
import config

model_path_7b = "/root/llm/Baichuan-7B"
model_path = model_path_7b  # 可修改
num_gpus = 2 #可修改
current_device_map = config.device_map_7b # 可修改

def count_token(response:str,last_time:float)  ->str: 
    
    tokens_length = len(tokenizer.encode(response))
    token_generated_speed = round(tokens_length/last_time,2)  #单位tokens/s
    if torch.cuda.is_available():
        gpu_nums = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
    speed = f"环境：{gpu_nums}张{device_name}，生成速度：{token_generated_speed} tokens/s"
    return  speed
    

tokenizer = AutoTokenizer.from_pretrained(model_path_7b,trust_remote_code=True)
device_map = config.distribute_layers(num_gpus,current_device_map)
model = AutoModelForCausalLM.from_pretrained(model_path_7b, device_map=device_map, trust_remote_code=True)
while True:
    prompt = input("问：")
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    pred = model.generate(**inputs, max_new_tokens=3000,repetition_penalty=1.1)
    last_time = time.perf_counter()- start_time  # 时间单位s

    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print(response)
    speed = count_token(response,last_time)
    print(speed)