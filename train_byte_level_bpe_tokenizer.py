
# 2025/7/14
# wenzhe

from datasets import load_dataset
import multiprocessing
import os
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    # normalizers,
)
from tqdm import tqdm

# Hugging Face Datasets 默认会把数据集下载到 ~/.cache/huggingface/datasets（没空间了），模型和tokenizer会下载到 ~/.cache/huggingface/hub
# 我们可以通过设置环境变量来改变这个路径
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'

num_proc = multiprocessing.cpu_count() // 2
print(f"Number of processes: {num_proc}")


# 统一文件路径
# corpus/dataset_name.txt
def save_dataset_to_txt(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            f.write(item["text"] + "\n")


# 1. preprocess datasets
# 其实就是把各个数据集的文本数据提取出来，放到一个txt文件中
# 我们的数据集包含网页爬取数据（C4），文章数据（维基百科），代码数据（Tiny Codes），书籍数据（Gutenberg），对话问答数据（Stack Exchange）等

# 把文件路径定义成常量吧
c4_txt_path = "corpus/c4.txt"
wikipedia_txt_path = "corpus/wikipedia.txt"
tiny_codes_txt_path = "corpus/tiny_codes.txt"
gutenberg_txt_path = "corpus/gutenberg.txt"
stack_exchange_txt_path = "corpus/stack_exchange.txt"


# 感觉这些函数还挺相似的，可以提取出来
# 有两个地方不同，一个是加载数据集的方式
# 一个是预处理的方式
# 我们只需要提供两个函数就行了


def preprocess_dataset(txt_path, load_dataset_func, preprocess_func):
    if os.path.exists(txt_path):
        print(f"{txt_path} already processed. Skipping...")
        return
    print(f"Processing {txt_path}...")

    print(f"Loading dataset of {txt_path}...")
    original_dataset = load_dataset_func()

    print(f"Loaded dataset with {len(original_dataset)} samples.")
    preprocessed_dataset = original_dataset.map(
        preprocess_func,
        remove_columns=original_dataset.column_names,  # 去掉数据集中原本的列，节省内存
        num_proc=num_proc,  # 使用多核处理
        desc=f"Preprocessing {txt_path}",
    )

    print(f"Saving preprocessed dataset to {txt_path}...")
    save_dataset_to_txt(dataset=preprocessed_dataset, filename=txt_path)


# 1.1 C4
# https://huggingface.co/datasets/allenai/c4
def preprocess_c4():
    preprocess_dataset(
        txt_path=c4_txt_path,
        load_dataset_func=lambda: load_dataset(
            path="allenai/c4",
            name="en",
            split="train[:10%]",
            # split="train[:1000]",
            download_mode="reuse_dataset_if_exists"
        ),
        preprocess_func=lambda example: {"text": example["text"].strip()},
    )


# 1.2 wikipedia
# https://huggingface.co/datasets/wikimedia/wikipedia
def preprocess_wikipedia():
    preprocess_dataset(
        txt_path=wikipedia_txt_path,
        load_dataset_func=lambda: load_dataset(
            path="wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            # split="train[:1000]",
            download_mode="reuse_dataset_if_exists"
        ),
        preprocess_func=lambda example: {"text": example["text"].strip()},
    )


# 1.3 codes
# https://huggingface.co/datasets/nampdn-ai/tiny-codes
def preprocess_tiny_codes():
    preprocess_dataset(
        txt_path=tiny_codes_txt_path,
        load_dataset_func=lambda: load_dataset(
            path="nampdn-ai/tiny-codes",
            split="train",
            # split="train[:1000]",
            download_mode="reuse_dataset_if_exists"
        ),
        preprocess_func=lambda example: {
            "text": f"{example['prompt']} {example['response']}".strip()
        },
    )


# 1.4 gutenberg
# https://huggingface.co/datasets/eminorhan/gutenberg_en
def preprocess_gutenberg():
    preprocess_dataset(
        txt_path=gutenberg_txt_path,
        load_dataset_func=lambda: load_dataset(
            path="eminorhan/gutenberg_en",
            name="chunk_size_1024",
            split="train",
            # split="train[:1000]",
            download_mode="reuse_dataset_if_exists"
        ),
        preprocess_func=lambda example: {"text": example["text"].strip()},
    )


# 1.5 stack exchange
# https://huggingface.co/datasets/donfu/oa-stackexchange
def preprocess_stack_exchange():
    preprocess_dataset(
        txt_path=stack_exchange_txt_path,
        load_dataset_func=lambda: load_dataset(
            path="donfu/oa-stackexchange",
            split="train",
            # split="train[:1000]",
            download_mode="reuse_dataset_if_exists"
        ),
        preprocess_func=lambda example: {
            "text": f"{example['INSTRUCTION']} {example['RESPONSE']}".strip()
        },
    )


# tokenizer

def train_byte_level_bpe_tokenizer():

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = None  # No normalization, default for GPT2
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)]
    )
    tokenizer.decoder = decoders.Sequence([decoders.ByteLevel()])

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        # min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # ← key point: includes all 256 bytes
    )

    print("Training tokenizer on datasets...")
    tokenizer.train(
        files=[
            c4_txt_path,
            wikipedia_txt_path,
            tiny_codes_txt_path,
            gutenberg_txt_path,
            stack_exchange_txt_path,
        ],
        trainer=trainer,
    )

    # 保存tokenizer
    print("Saving tokenizer ...")
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/byte_level_bpe_tokenizer_v1.json")


if __name__ == "__main__":
    # 1. preprocess datasets
    # preprocess_c4()
    # preprocess_wikipedia()
    # preprocess_tiny_codes()
    # preprocess_gutenberg()
    # preprocess_stack_exchange()

    # 2. train byte-level BPE tokenizer
    train_byte_level_bpe_tokenizer()

    # 3. done
    print("All tasks completed successfully!")