def get_latest_filename(dir):
    import os
    file_list = os.listdir(dir)
    file_list.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn)
                      if not os.path.isdir(dir + "/" + fn) else 0)
    
    filepath = os.path.join(dir,file_list[-1])

    return filepath

dir = "/root/llm/baichuan-13B/langchain-llm/papers"

file = get_latest_filename(dir)
print(file)