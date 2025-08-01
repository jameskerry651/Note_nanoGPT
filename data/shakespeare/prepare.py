import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
# 此代码行用于构建输入文件的完整路径。os.path.dirname(__file__) 获取当前脚本所在的目录路径，
# 然后使用 os.path.join 将该目录路径与 'input.txt' 文件名拼接起来，最终得到输入文件的完整路径。
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
# 计算数据长度
n = len(data)
# 划分训练集和验证集
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
# 使用之前初始化的 tiktoken 编码器 enc，对训练数据进行编码，将文本数据转换为 token ID 列表
train_ids = enc.encode_ordinary(train_data)
# 使用相同的编码器对验证数据进行编码，将文本数据转换为 token ID 列表
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
