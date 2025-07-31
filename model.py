"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """层归一化模块，支持可选的偏置参数。PyTorch原生LayerNorm不直接支持关闭偏置，此实现解决了该问题。"""
    
    def __init__(self, ndim, bias):
        """
        初始化层归一化模块。

        参数:
        ndim (int): 归一化的维度，通常为输入特征的维度。
        bias (bool): 是否使用偏置参数。True 表示使用，False 表示不使用。
        """
        super().__init__()
        # 初始化可训练的权重参数，形状为 (ndim,)，初始值全为 1。该权重用于对归一化后的输入进行缩放。
        self.weight = nn.Parameter(torch.ones(ndim))
        # 根据 bias 参数决定是否初始化可训练的偏置参数。
        # 若 bias 为 True，则初始化形状为 (ndim,) 且初始值全为 0 的偏置参数；
        # 若 bias 为 False，则不初始化偏置参数，将其设为 None。
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播方法，对输入进行层归一化操作。

        参数:
        input (torch.Tensor): 输入张量，需要进行层归一化的输入数据。
        返回:
        torch.Tensor: 经过层归一化处理后的张量。
        """
        # 使用 PyTorch 的 layer_norm 函数进行层归一化操作
        # input: 输入张量，即需要进行归一化处理的数据
        # self.weight.shape: 归一化的形状，使用权重的形状作为归一化的维度
        # self.weight: 可学习的权重参数，用于对归一化后的输入进行缩放
        # self.bias: 可学习的偏置参数，若初始化时未启用则为 None
        # 1e-5: 数值稳定性的小常数，防止分母为零
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """因果自注意力机制模块，确保序列中每个位置只能关注其前面的位置，防止信息泄露。支持Flash Attention加速（需PyTorch >= 2.0）。"""

    def __init__(self, config):
        """
        初始化因果自注意力模块。

        参数:
        config: 包含模型配置参数的对象，需包含以下属性：
            n_embd (int): 嵌入维度，即输入特征的维度。
            n_head (int): 注意力头的数量，需整除嵌入维度。
            block_size (int): 序列长度，用于生成因果掩码。
            dropout (float): dropout概率，用于正则化。
            bias (bool): 是否在线性层中使用偏置。
        """
        super().__init__()
        # 确保嵌入维度能被头数整除，保证每个头的维度一致
        assert config.n_embd % config.n_head == 0
        # 一次性对所有头的key、query、value进行线性投影，将输入维度映射到3倍的嵌入维度，nn.Linear 只对输入张量的最后一个维度进行操作
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 对自注意力的输出进行线性投影，将维度恢复到原始的嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 对注意力分数进行dropout操作，防止过拟合
        self.attn_dropout = nn.Dropout(config.dropout)
        # 对自注意力的输出进行dropout操作，防止过拟合
        self.resid_dropout = nn.Dropout(config.dropout)
        # 记录注意力头的数量
        self.n_head = config.n_head
        # 记录嵌入维度
        self.n_embd = config.n_embd
        # 记录dropout概率
        self.dropout = config.dropout
        # 检查当前PyTorch版本是否支持Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建因果掩码，确保每个位置只能关注其前面的位置
            # 使用下三角矩阵生成掩码，保证注意力只作用于输入序列的左侧
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # 获取输入张量的批次大小、序列长度和嵌入维度
        B, T, C = x.size()  # batch size = 12, sequence length = 1024, embedding dimensionality (n_embd = 768)
       
        # 通过线性投影计算所有头的query、key、value，并将其分割
        # 然后调整形状，将头的维度提前作为批次维度
        # 通过 self.c_attn 线性层对输入 x 进行投影，将输入维度映射到 3 倍的嵌入维度。
        # 然后使用 split 方法，沿着维度 2（特征维度）将投影结果分割为三个部分，
        # 分别赋值给查询（query）、键（key）和值（value）张量，每个部分的维度为嵌入维度 n_embd。

        # x = self.c_attn(x)的结果是 torch.Size([12, 1024, 2304])
        # 再将x 切割为 q, k, v  
        # q.size() = torch.Size([12, 1024, 768])
        # k.size() = torch.Size([12, 1024, 768])
        # v.size() = torch.Size([12, 1024, 768])
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) # 在维度2上切割，将2304切割为3个768
        # 将key的形状调整为 (B, nh, T, hs)，其中nh为头数，hs为每个头的维度
        # n_head = 12
        # k,view的结果是torch.Size([12, 1024, 12, 64]) ，一共12个头，每个头处理64维的嵌入向量
        # 将维度交换，调用transpose函数得到：
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # torch.Size([12, 12, 1024, 64])
        # 将query的形状调整为 (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 将value的形状调整为 (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 执行自注意力操作
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 使用Flash Attention的CUDA内核实现高效的注意力计算
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # 手动实现注意力机制
            # q = Tensor([12, 12, 1024, 64])
            # k.transpose = Tensor([12, 12, 64, 1024])
            # 计算query和key的点积，并进行缩放 k.size(-1) = 64
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 使用掩码，将掩码值为0的位置的注意力分数置为负无穷
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 对注意力分数进行softmax操作，得到注意力分布
            att = F.softmax(att, dim=-1)
            # 对注意力分布进行dropout操作
            att = self.attn_dropout(att) # att = Tensor([12, 12, 1024, 1024])
            # 根据注意力分布对value进行加权求和
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)  y = Tensor([12, 12, 1024, 64])
        # 将头的维度移回原来的位置，并重新拼接所有头的输出
        # contiguous() 函数用于确保张量在内存中是连续的，这在进行视图操作（如 view()）时是必要的。
        # y.transpose(1, 2) 会将维度 1 和 2 进行交换，结果的形状为 (B, T, nh, hs)= (12, 1024, 12, 64)。
        # view(B, T, C) 会将张量重新形状为 (B, T, C)，其中 C 是所有头的维度拼接后的结果。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # y = Tensor([12, 1024, 768])

        # 对自注意力的输出进行线性投影，并应用dropout
        y = self.resid_dropout(self.c_proj(y)) # y = Tensor([12, 1024, 768]) ,这里只是线性投影，没有激活函数，不改变嵌入维度
        return y

class MLP(nn.Module):
    """多层感知机模块，包含两个线性变换和GELU激活函数，用于Transformer块中的前馈网络部分。"""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer块模块，由层归一化、因果自注意力机制和多层感知机组成，包含残差连接。"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """GPT模型配置类，存储模型的超参数，如序列长度、词汇表大小、层数、头数和嵌入维度等。"""
    block_size: int = 1024 # 序列长度
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12 # block的数量
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    """GPT语言模型主类，包含完整的Transformer架构，支持文本生成和预训练权重加载。实现了前向传播、优化器配置和文本生成等核心功能。"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 使用 nn.ModuleDict 构建 Transformer 模块字典，方便管理模型的各个组件
        self.transformer = nn.ModuleDict(dict(
            # 词元嵌入层（Word Token Embedding），将输入的词元索引转换为词向量
            # config.vocab_size 为词汇表大小，config.n_embd 为嵌入维度
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入层（Word Position Embedding），为输入序列中的每个位置生成位置向量
            # config.block_size 为序列最大长度，config.n_embd 为嵌入维度
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout 层，用于在训练过程中随机将部分元素置为 0，防止过拟合
            # config.dropout 为 Dropout 概率
            drop = nn.Dropout(config.dropout),
            # 堆叠的 Transformer 块，使用 nn.ModuleList 管理多个 Block 实例
            # config.n_layer 为 Transformer 块的数量
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终的层归一化层（Final Layer Normalization），对 Transformer 块的输出进行归一化
            # config.n_embd 为归一化的维度，config.bias 决定是否使用偏置
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 线性层，将嵌入维度映射到词汇表大小
        # 使用权重绑定（weight tying）时，使用 torch.compile() 会产生一些警告：
        # "UserWarning: functional_call 被传入了多个绑定权重的值。
        # 此行为已被弃用，未来版本中将报错"
        # 目前还不完全清楚这是什么问题，到目前为止似乎没有危害。TODO: 调查此问题
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。
        对于非嵌入参数计数（默认情况），位置嵌入参数会被减去。
        原本词元嵌入参数也会被减去，但由于参数共享，这些参数实际上被用作最后一层的权重，因此我们将它们包含在内。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播函数，计算模型的输出和损失（如果提供了目标值）。
        参数：
        - idx: 输入的词元索引张量，形状为 (b, t)，其中 b 为批次大小，t 为序列长度。
        - targets: 可选的目标值张量，形状为 (b, t)，用于计算损失。
        返回：
        - logits: 模型的输出张量，形状为 (b, t, vocab_size)，其中 vocab_size 为词汇表大小。
        - loss: 如果提供了目标值，返回计算得到的损失值；否则返回 None。
        """
        device = idx.device # 获取输入张量的设备
        # 示例：如果 idx 是 [[10, 25, 4], [31, 19, 88]]，那么 b=2, t=3。
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 这个 `pos` 张量 (`[0, 1, 2, ..., t-1]`) 就是每个词元的位置标签
        pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状为 (t)

        # 前向传播GPT模型本身
        tok_emb = self.transformer.wte(idx) # 词元嵌入，形状为 (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状为 (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # x = Tensor of shape (b, t, n_embd)

        if targets is not None:
            # 如果提供了目标值，同时计算损失
            logits = self.lm_head(x) # 线性层，将嵌入维度映射到词汇表大小,形状为 (b, t, vocab_size), vocab_size: int = 50304 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时的小型优化：仅对最后一个位置进行lm_head前向传播
            logits = self.lm_head(x[:, [-1], :]) # 注意：使用列表 [-1] 以保留时间维度
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # 必要时对模型进行修改以减小块大小
        # 例如，我们可能加载了 GPT2 预训练模型检查点（块大小为 1024）
        # 但希望为一些更小、更简单的模型使用更小的块大小
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 以 A100 GPU 的 bfloat16 峰值浮点运算次数（FLOPS）为单位，估算模型的浮点运算利用率（MFU） """
        # 首先估算每次迭代的浮点运算次数。
        # 参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        给定一个条件索引序列 idx(形状为 (b,t) 的 LongTensor),
        该函数会根据条件序列生成 max_new_tokens 个新的标记，
        并将生成的标记反馈到模型中，以进行下一次预测。
        通常,在使用此函数时,模型应该处于评估模式(model.eval()）。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得过长，我们必须将其裁剪为 block_size 的长度
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播模型，获取序列中索引对应的 logits
            logits, _ = self(idx_cond)
            # 提取最后一步的 logits，并按所需的温度系数进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选操作：将 logits 裁剪为仅保留前 k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 函数，将 logits 转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样得到的索引追加到当前序列中，然后继续生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
