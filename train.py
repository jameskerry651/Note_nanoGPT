"""
此训练脚本既可以在单块 GPU 上以调试模式运行，
也可以在使用分布式数据并行 (DDP) 的大规模训练中运行。

在单块 GPU 上运行的示例：
$ python train.py --batch_size=32 --compile=False

在 1 个节点的 4 块 GPU 上使用 DDP 运行的示例：
$ torchrun --standalone --nproc_per_node=4 train.py

在 2 个节点的 4 块 GPU 上使用 DDP 运行的示例：
- 在第一个（主）节点上运行，示例 IP 为 123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果你的集群没有 Infiniband 互连，请在命令前添加 NCCL_IB_DISABLE=1）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 用于在 OpenWebText 上训练 gpt2 (124M) 的默认配置值
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = True # 如果为 True，每次评估后总是保存一个检查点
init_from = 'scratch' # 'scratch'（从头开始）、'resume'（从检查点恢复）或 'gpt2*'
# wandb 日志记录
wandb_log = False # 默认禁用
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# 数据
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的批量大小
batch_size = 12 # 如果 gradient_accumulation_steps > 1，这是微批量大小
block_size = 1024
# 模型
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # 预训练时设为 0 较好，微调时尝试 0.1 或更大的值
bias = False # 在 LayerNorm 和 Linear 层中是否使用偏置？
# adamw 优化器
learning_rate = 6e-4 # 最大学习率
max_iters = 600000 # 训练迭代的总次数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # 在此值处裁剪梯度，如果为 0.0 则禁用
# 学习率衰减设置
decay_lr = True # 是否衰减学习率
warmup_iters = 2000 # 预热的步数
lr_decay_iters = 600000 # 根据 Chinchilla 论文，应约等于 max_iters
min_lr = 6e-5 # 最小学习率，根据 Chinchilla 论文，应约等于 learning_rate/10
# DDP (Distributed Data Parallel) 设置，用于指定分布式训练时的通信后端
# 通信后端负责进程间的通信，常见的选项有 'nccl' 和 'gloo'
# 'nccl'（NVIDIA Collective Communications Library）是 NVIDIA GPU 上进行高效通信的首选后端，支持多 GPU 间的快速通信
# 'gloo' 是一个通用的通信后端，支持 CPU 和 GPU，通常用于调试或非 NVIDIA 硬件环境
# 此处选择 'nccl' 作为通信后端
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# 筛选出全局变量中所有不以 '_' 开头，且类型为 int、float、bool 或 str 的变量名
# 将这些变量名存储在 config_keys 列表中，这些变量后续会作为配置项使用
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# 执行 'configurator.py' 文件中的代码，该文件可能包含从命令行或配置文件中读取的配置
# 这些配置会覆盖当前脚本中已有的全局变量值
exec(open('configurator.py').read()) 
# 根据 config_keys 列表，从全局变量中提取对应的配置项及其值
# 将这些配置项和值存储在 config 字典中，该字典后续可用于日志记录等操作
config = {k: globals()[k] for k in config_keys} 
# -----------------------------------------------------------------------------

# 各种初始化、派生属性设置以及 I/O 设置
ddp = int(os.environ.get('RANK', -1)) != -1 # 当前是否在进行分布式数据并行 (DDP) 训练？
if ddp:
    # 初始化分布式进程组，使用指定的通信后端
    init_process_group(backend=backend)
    # 获取当前进程的全局排名，用于标识不同节点上的进程
    ddp_rank = int(os.environ['RANK'])
    # 获取当前进程在本地节点上的排名，用于指定使用的 GPU 设备
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # 获取参与分布式训练的进程总数
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # 设置当前进程使用的 GPU 设备
    device = f'cuda:{ddp_local_rank}'
    # 将当前进程的默认 CUDA 设备设置为指定的 GPU
    torch.cuda.set_device(device)
    # 判断当前进程是否为主进程，主进程负责日志记录、保存检查点等操作
    master_process = ddp_rank == 0 
    # 为每个进程设置不同的随机种子偏移量，确保训练的随机性
    seed_offset = ddp_rank 
    # 由于有 world_size 个进程同时进行训练，我们可以按比例缩小每个进程所需的梯度累积步数
    # 这里需要确保梯度累积步数能被进程总数整除
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是分布式训练，我们在单块 GPU 上运行，且只有一个进程
    master_process = True
    # 单进程训练时，随机种子偏移量为 0
    seed_offset = 0
    # 单进程训练时，进程总数为 1
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    # 如果是主进程，创建输出目录，exist_ok=True 表示如果目录已存在则不会报错
    os.makedirs(out_dir, exist_ok=True)
# 设置随机种子，加上 seed_offset 确保不同进程有不同的随机种子
torch.manual_seed(1337 + seed_offset)
# 允许在矩阵乘法中使用 TF32 数据类型，TF32 能在保持一定精度的同时提升计算速度
torch.backends.cuda.matmul.allow_tf32 = True 
# 允许在 cuDNN 中使用 TF32 数据类型，以提升计算性能
torch.backends.cudnn.allow_tf32 = True 
# 判断当前设备类型，如果设备名包含 'cuda' 则为 'cuda'，否则为 'cpu'，后续用于 torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu' 
# 注意：float16 数据类型会自动使用梯度缩放器（GradScaler）
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 如果使用 CPU 设备，则不进行自动混合精度训练；否则使用 torch.amp.autocast 进行自动混合精度训练
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # 从数据中随机抽取 batch_size 个样本的起始索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 生成 batch_size 个随机整数，范围在 [0, len(data) - block_size) 之间
    # 根据起始索引 ix 从数据中提取输入序列 x 和目标序列 y
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 固定数组 x 和 y，这样可以异步地将它们移动到 GPU 上（non_blocking=True）
        # .pin_memory(): 这是一个性能优化。它将张量数据锁在 CPU 的“固定内存”（Pinned Memory）中。被固定的内存可以更快地被复制到 GPU。
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    # 返回输入序列 x 和目标序列 y，其形状均为 (batch_size, block_size)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性保持一致，否则无法恢复训练
    # 其余属性（例如 dropout）可以保持命令行中指定的值
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复状态字典的键 :(
    # 说实话，不清楚检查点有时为何会有这个前缀，需要进一步调试
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI GPT-2 的权重初始化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读取创建的配置参数，以便我们能正确地将它们存储到检查点中
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，使用模型手术缩小模型的块大小
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 确保检查点中的值是正确的
model.to(device)

# 初始化一个梯度缩放器。如果 enabled=False，缩放器不执行任何操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# 将模型包装到 DDP 容器中
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 有助于使用多个批次对任一数据分割集估算任意精度的损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
