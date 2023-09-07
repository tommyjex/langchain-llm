import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from count import count_everything
import time
import config

st.set_page_config(page_title="AI助手")
st.title("你的AI助手")

model_path_13b = "/root/llm/baichuan-13B/Baichuan-13B-Chat"
model_path_7b = "/root/llm/Baichuan-7B"
model_path = model_path_13b  # 可修改
num_gpus = 1 #可修改
current_device_map = config.device_map_13b # 可修改

@st.cache_resource
def init_model():

    device_map = config.distribute_layers(num_gpus,current_device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map="auto",
        device_map = device_map,
        trust_remote_code=True
    )

    # # 量化int8
    # model = AutoModelForCausalLM.from_pretrained("/root/llm/baichuan-13B/Baichuan-13B-Chat",trust_remote_code=True,torch_dtype=torch.float16)
    # model = model.quantize(8).cuda()

    # 量化int4
    # model = AutoModelForCausalLM.from_pretrained("/root/llm/baichuan-13B/Baichuan-13B-Chat",trust_remote_code=True,torch_dtype=torch.float16)
    # model = model.quantize(4).cuda()
         
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是徐建华，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    messages = init_chat_history()
    model, tokenizer = init_model()


    counter = count_everything()
    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            start_time = time.perf_counter()
            # response = model.chat(tokenizer, messages, stream=False)
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            last_time = time.perf_counter()- start_time  # 时间单位s
            speed = counter.count_token(response,last_time)
            # placeholder.markdown(response)
            print(response)
        messages.append({"role": "assistant", "content": response})
        # print(json.dumps(messages, ensure_ascii=False), flush=True)
        print(speed)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
