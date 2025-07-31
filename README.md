
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

这是用于训练/微调中型 GPT 模型的最简单、最快速的代码仓库。它是对 [minGPT](https://github.com/karpathy/minGPT) 的重写，此处原文 “prioritizes teeth over education” 可能存在错误，推测原意为强调实用性（请检查原文表意）。目前该项目仍在积极开发中，但当前 `train.py` 文件可以在 OpenWebText 数据集上复现 GPT - 2 (124M) 模型，在单个配备 8 张 A100 40GB 显卡的节点上训练约 4 天即可完成。代码本身简洁易读：`train.py` 是一个约 300 行的样板训练循环，`model.py` 是一个约 300 行的 GPT 模型定义，还可以选择加载 OpenAI 的 GPT - 2 模型权重。就这么简单。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，你可以很容易地根据自己的需求进行修改，从头训练新模型，或者微调预训练的检查点（例如，当前可用的最大预训练起点模型是 OpenAI 的 13 亿参数的 GPT - 2 模型）。

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

如果你查看配置文件，就会发现我们正在训练一个上下文大小最大为 256 个字符、具有 384 个特征通道的 GPT 模型，它是一个 6 层的 Transformer，每层有 6 个注意力头。在一块 A100 GPU 上，这次训练大约需要 3 分钟，最佳验证损失为 1.4697。根据配置，模型检查点会被写入 `--out_dir` 指定的 `out-shakespeare-char` 目录。因此，一旦训练完成，我们可以通过将采样脚本指向该目录，从最佳模型中进行采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哈哈  `¯\_(ツ)_/¯`。对于一个在 GPU 上训练了 3 分钟的字符级模型来说，效果还不错。通过在这个数据集上微调预训练的 GPT - 2 模型，很可能会得到更好的结果（请参阅后面的微调部分）。

**我只有一台 MacBook**（或其他廉价电脑）。不用担心，我们仍然可以训练一个 GPT 模型，只是需要降低一些要求。我建议安装最新的 PyTorch 夜间版本（安装时可在[此处选择](https://pytorch.org/get-started/locally/)），因为它目前很有可能让你的代码运行得更高效。但即使不安装，一个简单的训练运行命令可能如下所示：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

在这里，由于我们使用的是 CPU 而非 GPU 进行运行，因此必须同时设置 `--device=cpu`，并使用 `--compile=False` 关闭 PyTorch 2.0 的编译功能。在评估时，我们会得到一个稍显嘈杂但速度更快的估计值（`--eval_iters=20`，从 200 下调而来），上下文大小仅为 64 个字符，而非 256 个，每次迭代的批量大小也只有 12 个样本，而非 64 个。我们还将使用一个小得多的 Transformer 模型（4 层，4 个注意力头，128 维嵌入大小），并将迭代次数减少到 2000 次（相应地，通常使用 `--lr_decay_iters` 将学习率衰减到最大迭代次数附近）。由于我们的网络规模很小，我们也会降低正则化强度（`--dropout=0.0`）。这样的配置仍然大约需要 3 分钟就能运行完成，但验证损失仅为 1.88，生成的样本质量也更差，但这仍然很有趣：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

对于在 CPU 上运行约 3 分钟的模型来说，能生成有一定风格的字符已经不错了。如果你愿意多等一会儿，可以随意调整超参数，增大网络规模、上下文长度（`--block_size`）、训练时长等。

最后，在配备 Apple Silicon 的 Macbook 上，使用较新的 PyTorch 版本时，请确保添加 `--device=mps`（“Metal Performance Shaders” 的缩写）；这样 PyTorch 会使用芯片内置的 GPU，这能 *显著* 加快训练速度（2 - 3 倍），并允许你使用更大的网络。更多信息请参阅 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## reproducing GPT-2

更专业的深度学习从业者可能对复现 GPT - 2 的结果更感兴趣。那么接下来——我们首先对数据集进行分词，在这个例子中是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，它是 OpenAI 私有 WebText 数据集的开源复现版本：

```sh
python data/openwebtext/prepare.py
```

这会下载并对 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集进行分词。它将创建一个 `train.bin` 和 `val.bin` 文件，这些文件以单个序列的形式保存 GPT2 的 BPE 标记 ID，并以原始的 uint16 字节格式存储。然后我们就可以开始训练了。要复现 GPT - 2 (124M) 模型，你至少需要一个配备 8 张 A100 40GB 显卡的节点，并运行以下命令：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这将使用 PyTorch 分布式数据并行（DDP）运行约 4 天，最终验证损失降至约 2.85。目前，仅在 OpenWebText（OWT）数据集上评估的 GPT - 2 模型的验证损失约为 3.11，但如果对其进行微调，损失会降至约 2.85（这是由于明显的领域差异），使得这两个模型的表现大致匹配。

如果你处于集群环境且拥有多个 GPU 节点，你可以让 GPU 火力全开，例如跨 2 个节点运行，如下所示：

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

对网络互连进行基准测试是个不错的主意（例如使用 iperf3）。特别地，如果你没有 Infiniband 网络，那么在上述启动命令前还需添加 `NCCL_IB_DISABLE=1`。多节点训练仍能运行，但速度很可能会非常 *缓慢*。默认情况下，检查点会定期写入 `--out_dir` 指定的目录。我们只需运行 `python sample.py` 即可从模型中采样。

最后，若要在单块 GPU 上进行训练，只需运行 `python train.py` 脚本。查看该脚本的所有参数，它设计得非常易读、易于修改且透明。你很可能需要根据自己的需求调整其中的一些变量。

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

然而，我们必须注意到，GPT - 2 是在（封闭的、从未公开过的）WebText 数据集上训练的，而 OpenWebText 只是对该数据集尽力复刻的开源版本。这意味着存在数据集领域差异。事实上，使用 GPT - 2 (124M) 检查点并在 OpenWebText 数据集上直接微调一段时间后，损失可降至约 2.85。就复现而言，这就成为了更合适的基准。

## finetuning

微调与训练并无本质区别，我们只需确保从预训练模型初始化，然后使用较小的学习率进行训练。若要了解如何在新文本上微调 GPT 模型，可前往 `data/shakespeare` 目录，运行 `prepare.py` 脚本以下载 Tiny Shakespeare 数据集，并使用 GPT - 2 的 OpenAI BPE 分词器将其转换为 `train.bin` 和 `val.bin` 文件。与 OpenWebText 不同，这个过程只需几秒钟即可完成。微调所需时间可能非常短，例如在单块 GPU 上仅需几分钟。运行以下示例进行微调：

```sh
python train.py config/finetune_shakespeare.py
```

这将加载 `config/finetune_shakespeare.py` 中的配置参数覆盖项（不过我并没有对这些参数进行太多调优）。基本上，我们使用 `init_from` 从 GPT2 检查点进行初始化，然后像正常训练一样进行训练，只是训练时间更短，学习率更小。如果你遇到内存不足的问题，可以尝试减小模型大小（模型大小选项有 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`），或者尝试减小 `block_size`（上下文长度）。最佳的检查点（验证损失最低的）将保存在 `out_dir` 目录下，例如，根据配置文件，默认会保存在 `out-shakespeare` 目录中。然后你可以运行 `sample.py --out_dir=out-shakespeare` 代码：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇哦，GPT，这生成内容有点暗黑风了。我其实没怎么调配置里的超参数，大家可以随意尝试调整！

## sampling / inference

使用脚本 `sample.py` 可以从 OpenAI 发布的预训练 GPT - 2 模型中采样，也可以从你自己训练的模型中采样。例如，以下是从可用的最大模型 `gpt2-xl` 中采样的方法：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果要从你自己训练的模型中采样，使用 `--out_dir` 指向代码适当的位置即可。你也可以使用文件中的一些文本作为提示，例如 ```python sample.py --start=FILE:prompt.txt```。

## 效率注意事项

对于简单的模型基准测试和分析，`bench.py` 脚本可能会很有用。它与 `train.py` 中的训练循环类似，只是省略了其他复杂性。
请注意，代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写本文时（2022 年 12 月 29 日），这使得 `torch.compile()` 在night版本中可用。这一行代码带来的性能提升十分显著，例如将每次迭代的时间从约 250 毫秒/次缩短至 135 毫秒/次。PyTorch 团队干得漂亮！

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

请注意，此仓库默认使用 PyTorch 2.0（即 `torch.compile`）。这是一项相当新的实验性功能，并非在所有平台上都可用（例如 Windows）。如果你遇到相关的错误信息，尝试添加 `--compile=False` 标志来禁用此功能。这会降低代码的运行速度，但至少代码能够运行。

若想了解有关此仓库、GPT 和语言建模的一些背景知识，观看我的 [从零到英雄系列](https://karpathy.ai/zero-to-hero.html) 可能会有所帮助。具体来说，如果你已有一些语言建模的基础，[GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) 很受欢迎。

更多问题/讨论，请随时加入 **#nanoGPT** 频道：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
