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

# 学习率衰减调度器（带热身阶段的余弦衰减策略）
# 此函数根据当前迭代次数 it 计算并返回对应的学习率。该策略包含三个阶段：
# 1. 热身阶段：在 warmup_iters 步内线性增加学习率
# 2. 衰减阶段：在 warmup_iters 到 lr_decay_iters 步之间使用余弦函数衰减学习率
# 3. 稳定阶段：在 lr_decay_iters 步之后保持最小学习率
def get_lr(it):
    # 1) 热身阶段：在 warmup_iters 步内线性增加学习率
    # 从 0 开始逐步增加到初始学习率 learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 如果当前迭代次数超过 lr_decay_iters，返回最小学习率 min_lr
    if it > lr_decay_iters:
        return min_lr
    # 3) 在热身阶段和衰减结束之间，使用余弦衰减策略将学习率降到最小学习率
    # 计算当前迭代次数在衰减阶段的比例，范围在 0 到 1 之间
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # 确保衰减比例在有效范围内
    assert 0 <= decay_ratio <= 1
    # 使用余弦函数计算衰减系数，范围在 0 到 1 之间
    # 随着迭代次数增加，系数从 1 逐渐减小到 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 范围 0..1
    # 根据衰减系数计算当前学习率，从初始学习率逐渐衰减到最小学习率
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环：不断迭代进行模型训练，直到满足终止条件
X, Y = get_batch('train')  # 获取第一个训练批次的数据，X 为输入序列，Y 为目标序列
t0 = time.time()  # 记录当前时间，用于后续计算每个迭代的耗时
local_iter_num = 0  # 记录当前进程生命周期内的迭代次数
# 如果使用了分布式数据并行（DDP），则通过 model.module 获取原始模型；否则直接使用 model
raw_model = model.module if ddp else model 
running_mfu = -1.0  # 初始化运行时的模型利用率（MFU），-1.0 表示尚未计算

while True:
    # 根据当前迭代次数确定本次迭代的学习率
    # 如果开启了学习率衰减（decay_lr 为 True），则调用 get_lr 函数计算学习率
    # 否则使用初始设置的学习率 learning_rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    # 遍历优化器的参数组，将计算得到的学习率应用到每个参数组中
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 评估模型在训练集和验证集上的损失，并在必要时保存模型检查点
    # 检查当前迭代次数是否达到评估间隔，并且当前进程是否为主进程
    # 只有在满足这两个条件时，才会进行模型评估和保存检查点等操作
    # 这样设计是为了避免多个进程重复进行评估和日志记录，减轻计算负担
    if iter_num % eval_interval == 0 and master_process:
        # 调用 estimate_loss 函数评估模型在训练集和验证集上的损失
        # 该函数会返回一个字典，包含 'train' 和 'val' 两个键，分别对应训练集和验证集的损失
        losses = estimate_loss()
        # 打印当前迭代次数下，模型在训练集和验证集上的损失
        # 保留四位小数，方便观察模型的训练效果
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 检查是否启用了 wandb 日志记录功能
        # 如果启用了，将当前迭代次数、训练集损失、验证集损失、学习率和模型利用率记录到 wandb 中
        if wandb_log:
            wandb.log({
                "iter": iter_num,  # 当前迭代次数
                "train/loss": losses['train'],  # 训练集损失
                "val/loss": losses['val'],  # 验证集损失
                "lr": lr,  # 当前学习率
                "mfu": running_mfu*100, # 将模型利用率转换为百分比后记录
            })

        # 检查当前验证集损失是否小于之前记录的最佳验证集损失，或者是否设置了总是保存检查点
        # 如果满足任一条件，则更新最佳验证集损失，并保存模型检查点
        if losses['val'] < best_val_loss or always_save_checkpoint:
            # 更新最佳验证集损失为当前验证集损失
            best_val_loss = losses['val']
            # 确保当前迭代次数大于 0，避免在初始迭代时保存空的检查点
            if iter_num > 0:
                # 创建一个字典，包含模型的状态字典、优化器的状态字典、模型参数、当前迭代次数、最佳验证集损失和配置信息
                # 这些信息将用于后续恢复训练
                checkpoint = {
                    'model': raw_model.state_dict(),  # 模型的状态字典，包含模型的所有参数
                    'optimizer': optimizer.state_dict(),  # 优化器的状态字典，包含优化器的所有参数
                    'model_args': model_args,  # 模型的配置参数
                    'iter_num': iter_num,  # 当前迭代次数
                    'best_val_loss': best_val_loss,  # 最佳验证集损失
                    'config': config,  # 训练配置信息
                }
                # 打印保存检查点的信息，提示用户当前正在保存检查点
                print(f"saving checkpoint to {out_dir}")
                # 将检查点字典保存到指定的文件中
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # 检查当前是否为第一次迭代，并且是否设置了仅评估模式
    # 如果满足这两个条件，说明用户只想评估模型，不进行训练，因此跳出训练循环
    if iter_num == 0 and eval_only:
        break

    # 前向传播、反向传播和参数更新，可选择使用梯度累积来模拟更大的批量大小
    # 如果数据类型为 float16，则使用梯度缩放器
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 在 DDP 训练中，我们只需要在最后一个微步同步梯度。
            # 官方的做法是使用 model.no_sync() 上下文管理器，但是
            # 我不太喜欢这种做法，因为它会让代码变得臃肿，还迫使我们重复编写代码
            # 查看该上下文管理器的源码，它只是切换了这个变量的值
            # 因此，我选择手动设置这个变量的值，以实现相同的效果
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 缩放损失以考虑梯度累积
        # 当模型在 GPU 上进行前向传播时，立即异步预取下一批数据
        X, Y = get_batch('train')
        # 反向传播，如果使用 fp16 训练则进行梯度缩放
        scaler.scale(loss).backward()
    # 裁剪梯度
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 如果使用 fp16 进行训练，更新优化器和梯度缩放器
    scaler.step(optimizer)
    scaler.update()
    # 尽快清除梯度，不再需要这些内存
    optimizer.zero_grad(set_to_none=True)

    # 计时和日志记录
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 获取浮点数格式的损失值。注意：这是一个 CPU-GPU 同步点
        # 放大损失值以抵消上面的除法操作，近似得到真实的总损失（精确值应该是求和）
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让训练循环稍微稳定一下
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
