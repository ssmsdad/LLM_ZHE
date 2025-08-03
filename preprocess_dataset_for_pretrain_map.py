
# 2025/7/16
# wenzhe


import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from tokenizers import Tokenizer
import multiprocessing

block_size = 1024
output_dir = "./pretrain_data_chunks"
# num_proc 控制 map 和 flat_map 的并行进程数
num_proc = max(1, multiprocessing.cpu_count() // 2) # 建议使用 CPU 核数的一半，但对于低内存环境，可能需要设置为更小的值
# num_proc = 1 # 如果内存非常吃紧，可以先设置为1进行测试

def load_tokenizer(tokenizer_path="tokenizer/byte_level_bpe_tokenizer_v1.json"):
    """加载自定义分词器并返回PreTrainedTokenizerFast对象"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)
    # 使用HuggingFace的PreTrainedTokenizerFast加载自定义的分词器，使其与training兼容
    # 后来发现使用AutoTokenizer.from_pretrained也完全可以，真是小丑了
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=hf_tokenizer)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer

def load_and_prepare_datasets(cache_dir="/data2/huggingface/datasets", prepared_path="pretrain_data_chunks/prepared_dataset.arrow"):
    """加载五个数据集并拼接、打乱"""
    os.makedirs(os.path.dirname(prepared_path), exist_ok=True) # 确保父目录存在
    if os.path.exists(prepared_path):
        print(f"直接加载缓存数据集: {prepared_path}")
        # load_from_disk 会加载整个数据集，可能内存消耗大
        # 对于非常大的数据集，可以考虑使用 streaming=True 如果只进行迭代
        dataset = Dataset.load_from_disk(prepared_path)
        return dataset
    
    print("首次运行，加载并拼接数据集...")
    c4 = load_dataset("allenai/c4", name="en", split="train[:10%]", cache_dir=cache_dir)
    wikipedia = load_dataset("wikimedia/wikipedia", name="20231101.en", split="train", cache_dir=cache_dir)
    tiny_codes = load_dataset("nampdn-ai/tiny-codes", split="train", cache_dir=cache_dir)
    gutenberg = load_dataset("eminorhan/gutenberg_en", name="chunk_size_1024", split="train", cache_dir=cache_dir)
    stack_exchange = load_dataset("donfu/oa-stackexchange", split="train", cache_dir=cache_dir)
    
    dataset = concatenate_datasets([c4, wikipedia, tiny_codes, gutenberg, stack_exchange])
    dataset = dataset.shuffle(seed=42) 
    
    print(f"保存拼接后的数据集到: {prepared_path}")
    # save_to_disk 会将整个数据集序列化到磁盘，这在第一次运行后很有用
    dataset.save_to_disk(prepared_path)
    print("数据集拼接并保存完成。")
    return dataset

# 它的核心思想是：接收一个批次的原始文本，为这个批次中的每一个原始文本tokenize并划分它所有的块
def tokenize_and_chunk_for_flat_map(examples, tokenizer, block_size, max_text_len=4096, eos_token="</s>"):
    input_ids_batch = []
    attention_mask_batch = []

    for text in examples["text"]:
        # if len(text) > max_text_len:
        #     continue # 过滤掉过长的文本
        if not isinstance(text, str):
            # 跳过 None 或非字符串类型
        # input_ids_batch.append([])
        # attention_mask_batch.append([])
            continue
        text_with_eos = text + eos_token

        # 对单个文本进行分词
        # 注意：这里我们对每个文本单独分词，而不是对整个批次一次性分词，这样可以更好地控制内存
        tokenized_output_single = tokenizer(text_with_eos, add_special_tokens=False)
        
        # 获取分词后的input_ids和attention_mask (它们已经是列表了)
        input_ids_single = tokenized_output_single["input_ids"]
        attention_mask_single = tokenized_output_single["attention_mask"]

        # 保留最后一块不足 block_size 的部分，并 pad 到 block_size，mask 用 0 补齐，不浪费样本
        blocks = []
        masks = []
        total_length = len(input_ids_single)
        for i in range(0, total_length, block_size):
            block = input_ids_single[i:i+block_size]
            mask = attention_mask_single[i:i+block_size]
            if len(block) < block_size:
                pad_len = block_size - len(block)
                block = block + [tokenizer.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            blocks.append(block)
            masks.append(mask)
        input_ids_batch.append(blocks)
        attention_mask_batch.append(masks)
    
    return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

def flatten_chunks(examples):
    # examples["input_ids"] 是一条数据，每个元素是一个 block列表
    # 需要展平成一维
    flat_input_ids = []
    flat_attention_mask = []
    for blocks, masks in zip(examples["input_ids"], examples["attention_mask"]):
        flat_input_ids.extend(blocks)
        flat_attention_mask.extend(masks)
    return {"input_ids": flat_input_ids, "attention_mask": flat_attention_mask}

if __name__ == "__main__":
    print(f"Starting preprocessing with {num_proc} processes.")
    
    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded.")

    dataset = load_and_prepare_datasets()
    print(f"原始数据集大小: {len(dataset)} 条")
    
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # 使用 map 进行初步处理：分词和分块
    # 注意：这里 `map` 返回的 `processed_dataset` 的每行仍然对应原始数据集的每行
    # 但是每行的 "input_ids" 和 "attention_mask" 将是列表的列表（包含该原始文本的所有块）
    print("开始分词和分块...")
    processed_dataset = dataset.map(
        lambda examples: tokenize_and_chunk_for_flat_map(examples, tokenizer, block_size),
        batched=True, # 处理批次数据
        num_proc=num_proc, # 使用多进程
        remove_columns=[col for col in dataset.column_names if col != "input_ids" and col != "attention_mask"], # 移除原始文本列及其他非必要的列
        load_from_cache_file=False, # 调试时设为False，正式运行时可以改为True利用缓存
        desc="Tokenizing and chunking dataset"
    )
    print(f"初步处理后的数据集 (processed_dataset) 大小: {len(processed_dataset)} 条 (行数不变)")
    '''此时 processed_dataset 的结构示例:
    [
        {
            "input_ids": [
                [101, 102, ..., 1124],      # 第1条文本的第1个block
                [1125, 1126, ..., 2148],    # 第1条文本的第2个block
            ],
            "attention_mask": [
                [1, 1, ..., 1],             # 第1个block的mask
                [1, 1, ..., 1],             # 第2个block的mask
            ]
        },
        {
            "input_ids": [
                [201, 202, ..., 1225],      # 第2条文本的唯一一个block
            ],
            "attention_mask": [
                [1, 1, ..., 1],             # mask
            ]
        },
        {
            "input_ids": [
                [301, 302, ..., 1324],      # 第3条文本的第1个block
                [1325, 1326, ..., 2348],    # 第3条文本的第2个block
                [2349, 2350, ..., 3372],    # 第3条文本的第3个block
            ],
            "attention_mask": [
                [1, 1, ..., 1],
                [1, 1, ..., 1],
                [1, 1, ..., 1],
            ]
        }
    ]
    '''

    print("开始展平数据集...")
    final_dataset = processed_dataset.map(
        flatten_chunks,
        batched=True,
        num_proc=num_proc,
        desc="Flattening chunks"
    )
    print(f"最终分块后的数据集 (final_dataset) 大小: {len(final_dataset)} 条 (总块数)")
    '''此时 final_dataset 的结构示例:
    [
        {"input_ids": [101, 102, ..., 1124], "attention_mask": [1, 1, ..., 1]},      # 第1条文本的第1个block
        {"input_ids": [1125, 1126, ..., 2148], "attention_mask": [1, 1, ..., 1]},    # 第1条文本的第2个block
        {"input_ids": [201, 202, ..., 1225], "attention_mask": [1, 1, ..., 1]},      # 第2条文本的唯一一个block
        # ...后面是所有文本的所有 block，全部展平成一行一个 block
    ]
    '''

    # 将最终的扁平化数据集保存为 Parquet 文件，datasets.to_parquet 会自动将大型数据集分割成多个文件
    tokenized_output_dir = os.path.join(output_dir, "tokenized_dataset")
    os.makedirs(tokenized_output_dir, exist_ok=True)
    print(f"将tokenized的数据集保存到: {tokenized_output_dir}/train_chunks_*.parquet")
    final_dataset.to_parquet(os.path.join(tokenized_output_dir, "train_chunks_*.parquet"))
    print("所有分块已保存为parquet文件！")