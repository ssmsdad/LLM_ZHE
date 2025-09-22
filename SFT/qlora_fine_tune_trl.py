# QLoRA Fine-tuning using TRL SFTTrainer
# 2025/7/30
# wenzhe

# 这是一个基于TRL库的SFTTrainer的QLoRA微调脚本
# SFTTrainer 与传统的Trainer相比，有以下几点优势：
# 1. 自动处理分词格式化（只传文本就可以了，不需要加进行额外的处理）与标签生成
# 2. 支持packing优化：可以自动按长度分组，减少填充，提高训练效率。

import os
# 设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
os.environ['WANDB_PROJECT'] = 'qlora-chatbot-finetune-trl'
import random
from datetime import datetime
from typing import Dict, List, Optional
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import wandb
import multiprocessing

num_proc = max(1, multiprocessing.cpu_count() // 2)


def format_conversation(conversation: List[Dict]) -> str:
    """将对话格式化为训练文本"""
    formatted_text = ""
    for turn in conversation:
        if turn['role'] == 'user':
            formatted_text += f"User: {turn['content']}\n"
        elif turn['role'] == 'assistant':
            formatted_text += f"Assistant: {turn['content']}\n"
    return formatted_text.strip()


def process_chatbot_arena_dataset(dataset, tokenizer, max_length: int = 1024) -> Dataset:
    """处理Chatbot Arena数据集"""
    
    def process_batch(batch):
        """批处理聊天机器人竞技场数据"""
        processed_texts = []
        processed_lengths = []
        
        for i in range(len(batch['winner'])):
            # 选择获胜的对话
            if batch['winner'][i] == 'model_a':
                conversation = batch['conversation_a'][i]
            elif batch['winner'][i] == 'model_b':
                conversation = batch['conversation_b'][i]
            else:  # tie的情况，随机选择一个
                conversation = random.choice([batch['conversation_a'][i], batch['conversation_b'][i]])
            
            # 只处理英文对话
            if batch['language'][i] == 'English' and conversation:
                formatted_text = format_conversation(conversation)
                
                # 检查长度是否合适
                tokens = tokenizer.encode(formatted_text)
                if len(tokens) <= max_length and len(tokens) > 20:
                    processed_texts.append(formatted_text)
                    processed_lengths.append(len(tokens))
        
        return {
            'text': processed_texts,
            'length': processed_lengths
        }
    
    # 使用map批处理
    print("处理数据集...")
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,  # 每批处理1000个样本
        remove_columns=dataset.column_names,  # 删除原始列
        desc="Processing Chatbot Arena dataset",  # 显示进度条
        num_proc=num_proc,  # 使用多进程加速
    )
    
    # 过滤掉空的批次
    processed_dataset = processed_dataset.filter(
        lambda x: len(x['text']) > 0,
        desc="Filtering empty batches"
    )
    
    print(f"处理完成，共有 {len(processed_dataset)} 条有效对话")
    return processed_dataset

# 如果使用transformers的Trainer，则需要手动处理文本和标签，我这里是没有用的
# def tokenize_function(self, examples):
#     """分词函数"""
#     # 在文本末尾添加EOS token </s>
#     texts = [text + self.tokenizer.eos_token for text in examples["text"]]
    
#     #这里的tokenized是一个batch的结果
#     tokenized = self.tokenizer(
#         texts,
#         truncation=True,
#         padding=False,
#         max_length=self.max_seq_length,
#         return_overflowing_tokens=False,    # 超过maxseq_length的文本将被丢弃
#     )
    
#     # 对于因果语言模型，labels应该是input_ids向右移动一位
#     # input_ids: [token1, token2, token3, ..., tokenN]
#     # labels:    [token2, token3, ..., tokenN, EOS]
#     labels = []
#     for input_ids in tokenized["input_ids"]:
#         # 创建向右移动一位的labels
#         label = input_ids[1:] + [self.tokenizer.eos_token_id]
#         labels.append(label)
    
#     tokenized["labels"] = labels
    
#     return tokenized

class QLoRATrainerTRL:
    """基于TRL SFTTrainer的QLoRA微调训练器"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_path: str = "../tokenizer/byte_level_bpe_tokenizer_v1.json",
        output_dir: str = "./sft_output",
        max_seq_length: int = 1024,
    ):
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.setup_model_and_tokenizer()
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        print(f"加载模型: {self.model_name}")
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 配置4bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # 改为float16避免数据类型不匹配
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            # device_map={'': torch.cuda.current_device()},  # 4bit量化必须指定单一设备
            device_map={"": local_rank},   # <---- 关键
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 明确数据类型
        )
        
        # 加载分词器 - 使用自定义的字节级BPE分词器，确保设置了所有必要的特殊token
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", 
            pad_token="<pad>", 
            eos_token="</s>", 
            unk_token="<unk>",
        )
        
        # 确保模型config与分词器token设置一致
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 验证特殊token
        print(f"分词器配置: pad_token_id={self.tokenizer.pad_token_id}, vocab_size={self.tokenizer.vocab_size}")
        if self.tokenizer.pad_token_id >= self.tokenizer.vocab_size:
            raise ValueError(f"Pad token ID超出词汇表范围: {self.tokenizer.pad_token_id} >= {self.tokenizer.vocab_size}")

        '''
        prepare_model_for_kbit_training：
        1、冻结模型参数，防止误更新。
        2、处理量化模型的梯度计算、输入输出等兼容性问题。
        3、确保量化权重和 LoRA adapter 能正确协同训练。
        '''
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA - 针对0.1B小模型优化
        lora_config = LoraConfig(
            r=8,  # 小模型使用较小的rank，避免过参数化
            lora_alpha=16,  # 相应减小alpha scaling
            target_modules=[
                "c_attn", "c_proj"  # GPT-2注意力层
            ],
            lora_dropout=0.05,  # 小模型降低dropout，保持学习能力
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # get_peft_model：把 LoRA adapter 层“插入”到原始大模型中
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def formatting_func(self, example):
        """
        TRL SFTTrainer的格式化函数（当前未使用）
        如果需要复杂的数据预处理，可以使用此函数替代dataset_text_field
        
        使用方法：
        - 如果数据需要复杂格式化：使用formatting_func参数
        - 如果数据字段简单直接：使用dataset_text_field参数（当前使用）
        """
        return example["text"]

    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        use_wandb: bool = True,
        packing: bool = False,
    ):
        """开始训练"""
        
        if use_wandb:
            wandb.init(
                project="qlora-chatbot-finetune-trl",
                name=f"qlora-trl-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": self.model_name,
                    "num_train_epochs": num_train_epochs,
                    "learning_rate": learning_rate,
                    "batch_size": per_device_train_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_seq_length": self.max_seq_length,
                    "packing": packing,
                }
            )
            
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset is not None else None,
            eval_strategy=("steps" if eval_dataset is not None else "no"),
            save_strategy=("steps" if eval_dataset is not None else "no"),
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            report_to="wandb" if use_wandb else "none",
            run_name=f"qlora-trl-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            max_length=self.max_seq_length,
            dataset_text_field="text",
            packing=packing,
            fp16=False,  # 启用fp16与量化配置匹配
            bf16=False,  # 禁用bf16
        )
            

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        # 开始训练
        print("开始TRL SFT训练...")
        trainer.train()
        
        # 在模型被get_peft_model包装后，trainer.save_model()只保存LoRA适配器
        # 因为我后边还要用这个模型来训练奖励模型并且作为强化学习微调的策略模型，所以我选择了合并LORA权重,直接加载更方便
        print("保存LoRA适配器...")
        trainer.save_model()
        
        # 保存合并后的完整模型（推荐用于后续使用）
        print("合并并保存完整模型...")
        merged_model_dir = f"{self.output_dir}_merged"
        os.makedirs(merged_model_dir, exist_ok=True)
        
        # 合并LoRA权重到基础模型
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(merged_model_dir)
        # self.tokenizer.save_pretrained(merged_model_dir)
        
        print(f"完整模型已保存到: {merged_model_dir}")
        
        if use_wandb:
            wandb.finish()
        
        return trainer

def main():
    """主函数"""
    print("开始基于TRL的QLoRA微调...")

    # 1. 加载分词器
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"
    print(f"加载分词器: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>", 
        pad_token="<pad>", 
        eos_token="</s>",
        unk_token="<unk>"
    )
    
    # 2. 加载并处理数据集
    print("加载Chatbot Arena数据集...")
    ds = load_dataset("lmsys/chatbot_arena_conversations")
    dataset = ds['train']
    processed_dataset = process_chatbot_arena_dataset(dataset, tokenizer, max_length=1024)
    
    # 3. 分割数据集
    dataset_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # # 4. 检查一些样本（可选）
    # print("\n样本数据预览:")
    # for i in range(min(2, len(train_dataset))):
    #     print(f"样本 {i+1}:")
    #     print(train_dataset[i]['text'][:200] + "...")
    #     print("-" * 50)
    
    # 5. 初始化TRL训练器
    model_name = "../output"  # 使用本地预训练完成的模型
    
    trainer = QLoRATrainerTRL(
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        output_dir="./sft_output",
        max_seq_length=1024,
    )
    
    # 6. 开始训练
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=2,  # 🔧 减少训练轮数，避免过拟合
        learning_rate=1e-4,  # 🔧 降低学习率，更温和的微调
        per_device_train_batch_size=1,  # 避免padding问题，但通过多GPU并行提高吞吐
        gradient_accumulation_steps=16,  # 保持有效batch size
        warmup_steps=100,  # 🔧 减少warmup步数
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        use_wandb=False,  # 设置为True如果要使用wandb
        packing=False,  # 禁用packing，避免序列长度问题
    )
    
    print("SFT训练完成！")
    print(f"LoRA适配器已保存到: {trainer.output_dir}")
    print(f"完整模型已保存到: {trainer.output_dir}_merged")
    

if __name__ == "__main__":
    main()
