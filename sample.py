"""
从训练好的模型中生成样本
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # 可以是'resume'（从out_dir加载）或gpt2变体（例如'gpt2-xl'）
out_dir = 'out' # 如果init_from不是'resume'，则此参数被忽略
start = "\n" # 或者" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # 要生成的样本数量
max_new_tokens = 500 # 每个样本生成的token数量
temperature = 0.8 # 1.0表示无变化，<1.0表示随机性降低，>1.0表示随机性增加（用于预测）
top_k = 200 # 只保留top_k个最可能的token，其他token的概率设为0
seed = 1337
device = 'cuda' # 例如：'cpu', 'cuda', 'cuda:0', 'cuda:1'等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32'或'bfloat16'或'float16'
compile = False # 使用PyTorch 2.0编译模型以提高速度
exec(open('configurator.py').read()) # 从命令行或配置文件覆盖参数
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 允许在矩阵乘法中使用tf32
torch.backends.cudnn.allow_tf32 = True # 允许在cudnn中使用tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后续torch.autocast使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型
if init_from == 'resume':
    # 从特定目录中保存的模型初始化
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 从给定的GPT-2模型初始化
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要PyTorch 2.0（可选）

# 检查数据集文件夹中是否存在元数据pickle文件
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 旧的检查点可能没有这些信息...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"从{meta_path}加载元数据...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO 希望能更通用地支持任意的编码器/解码器方案
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # 好的，默认假设使用gpt-2的编码方式
    print("未找到meta.pkl，假设使用GPT-2编码方式...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

# 编码提示词的开头部分
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 运行生成过程
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
