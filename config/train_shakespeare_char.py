# 训练一个微型的字符级莎士比亚模型
# 适合在 MacBook 等设备上进行调试和玩耍

out_dir = 'out-shakespeare-char'
eval_interval = 250 # 保持频繁，因为我们会过拟合
eval_iters = 200
log_interval = 10 # 不要过于频繁地打印

# 我们预计在这个小数据集上会过拟合，所以只在验证集表现提升时保存模型
always_save_checkpoint = False

wandb_log = False # 如果你愿意，可以通过命令行覆盖此设置
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # 最多考虑前 256 个字符的上下文

# 小型 GPT 模型 :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # 对于小型网络，可以将学习率设置得稍高一些
max_iters = 5000
lr_decay_iters = 5000 # 通常与 max_iters 设置为相同的值
min_lr = 1e-4 # 通常为 learning_rate 的 1/10
beta2 = 0.99 # 稍微增大该值，因为每次迭代的 token 数量较少

warmup_iters = 100 # 可能不是非常必要

# 在 MacBook 上还需要添加
# device = 'cpu'  # 仅在 CPU 上运行
# compile = False # 不使用 torch 编译模型
