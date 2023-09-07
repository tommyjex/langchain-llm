from transformers import AutoTokenizer
import torch


class count_everything():

    model_path = "/root/llm/baichuan-13B/Baichuan-13B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)



    # 计算token生成速度
    def count_token(self,response:str,last_time:float)  ->str: 
        
        tokens_length = len(self.tokenizer.encode(response))
        token_generated_speed = round(tokens_length/last_time,2)  #单位tokens/s
        if torch.cuda.is_available():
            gpu_nums = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
        speed = f"环境：{gpu_nums}张{device_name}，生成速度：{token_generated_speed} tokens/s"
        return  speed
    

    # def load_on_multi_gpus():
        