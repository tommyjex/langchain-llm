import math

# bichuan-7b 
device_map_7b={
        'model.embed_tokens': 0, 
        'model.layers.0': 0, 
        'model.layers.1': 0, 
        'model.layers.2': 0, 
        'model.layers.3': 0, 
        'model.layers.4': 0, 
        'model.layers.5': 0, 
        'model.layers.6': 0, 
        'model.layers.7': 1, 
        'model.layers.8': 1, 
        'model.layers.9': 1, 
        'model.layers.10': 1, 
        'model.layers.11': 1, 
        'model.layers.12': 1, 
        'model.layers.13': 1, 
        'model.layers.14': 1, 
        'model.layers.15': 1, 
        'model.layers.16': 2, 
        'model.layers.17': 2, 
        'model.layers.18': 2, 
        'model.layers.19': 2, 
        'model.layers.20': 2, 
        'model.layers.21': 2, 
        'model.layers.22': 2, 
        'model.layers.23': 2, 
        'model.layers.24': 2, 
        'model.layers.25': 3, 
        'model.layers.26': 3, 
        'model.layers.27': 3, 
        'model.layers.28': 3, 
        'model.layers.29': 3, 
        'model.layers.30': 3, 
        'model.layers.31': 3, 
        'model.norm': 3, 
        'lm_head': 3
        }

# baichuan 13b

device_map_13b = {'model.embed_tokens': 0, 
             'model.layers.0': 0, 
             'model.layers.1': 0, 
             'model.layers.2': 0, 
             'model.layers.3': 0, 
             'model.layers.4': 0, 
             'model.layers.5': 0, 
             'model.layers.6': 0, 
             'model.layers.7': 0, 
             'model.layers.8': 0, 
             'model.layers.9': 1, 
             'model.layers.10': 1, 
             'model.layers.11': 1, 
             'model.layers.12': 1, 
             'model.layers.13': 1, 
             'model.layers.14': 1, 
             'model.layers.15': 1, 
             'model.layers.16': 1, 
             'model.layers.17': 1, 
             'model.layers.18': 1, 
             'model.layers.19': 1, 
             'model.layers.20': 2, 
             'model.layers.21': 2, 
             'model.layers.22': 2, 
             'model.layers.23': 2, 
             'model.layers.24': 2, 
             'model.layers.25': 2, 
             'model.layers.26': 2, 
             'model.layers.27': 2, 
             'model.layers.28': 2, 
             'model.layers.29': 2, 
             'model.layers.30': 2, 
             'model.layers.31': 3, 
             'model.layers.32': 3, 
             'model.layers.33': 3, 
             'model.layers.34': 3, 
             'model.layers.35': 3, 
             'model.layers.36': 3, 
             'model.layers.37': 3, 
             'model.layers.38': 3, 
             'model.layers.39': 3, 
             'model.norm': 3, 
             'lm_head': 3}

# llama2-7b
device_map_of_llama2_7b = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 1}

# 把权重的所有层分布到指定数量的GPU
def distribute_layers(num_gpus:int,device_map:dict):
    
    layers_per_gpu = math.ceil(len(device_map)/num_gpus) 
    for index,key in enumerate(device_map):
        device_map[key] = math.floor(index/layers_per_gpu)
    return device_map

# example
# device_map = distribute_layers(3,device_map_13b)
# print(device_map)
