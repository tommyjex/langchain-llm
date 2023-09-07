from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import uvicorn,json
import torch


model_path = "/root/llm/baichuan-13B/Baichuan-13B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_path)

# 量化int8
# model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16)
# model = model.quantize(8).cuda()
# model.generation_config = GenerationConfig.from_pretrained(model_path)

# 量化int4
# model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16)
# model = model.quantize(4).cuda()
# model.generation_config = GenerationConfig.from_pretrained(model_path)

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    query = await request.json() #json对象的字典表示形式
    messages = [query]

    response = model.chat(tokenizer, messages)
    print(response)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return response


if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)