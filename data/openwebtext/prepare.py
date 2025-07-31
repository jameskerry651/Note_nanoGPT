# 将 openwebtext 数据集保存为二进制文件以便训练。以下链接提供了帮助：
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface 数据集

# .map() 调用中的工作进程数量
# 合适的数量大约是 CPU 核心数 // 2
num_proc = 8

# load_dataset() 调用中的工作进程数量
# 最佳数量可能与上面的 num_proc 不同，因为它还取决于网络速度。
# 不过通常比 1 要好
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # 在 huggingface 的 .cache 目录中占用 54GB 空间，约 800 万份文档 (8,013,769)
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # owt 默认只包含 'train' 划分，因此创建一个测试划分
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # 将测试划分重命名为 val

    # 结果如下：
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # 我们现在想对数据集进行分词。首先定义编码函数 (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary 会忽略任何特殊标记
        ids.append(enc.eot_token) # 添加文本结束标记，例如 gpt2 bpe 的标记为 50256
        # 注意：我认为 eot 应该前置而不是后置... 嗯。不过它被称为 "eot"（文本结束）...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # 对数据集进行分词
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="对划分进行分词",
        num_proc=num_proc,
    )

    # 将每个数据集中的所有 id 连接成一个大文件，供训练使用
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (可以这样做，因为 enc.max_token_value == 50256 小于 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'写入 {filename}'):
            # 将样本分批以加快写入速度
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # 写入内存映射文件
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin 约为 17GB，val.bin 约为 8.5MB
    # 训练集约有 90 亿个标记 (9,035,582,198)
    # 验证集约有 400 万个标记 (4,434,897)

    # 之后读取二进制文件，例如使用 numpy：
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
