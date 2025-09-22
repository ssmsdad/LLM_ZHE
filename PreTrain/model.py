
# 分布式训练主流程，支持4卡3060，DeepSpeed自动并行加速
# 2025/7/23
# wenzhe

import os
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
# 设置 PyTorch CUDA 内存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast,EarlyStoppingCallback,TrainerCallback
from datasets import Dataset,config
import glob
import logging
import torch
import time

logging.basicConfig(
    filename="train.log", 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # 追加模式，保留历史日志
)

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.last_log_time = None
    
    def _is_main_process(self):
        """检查是否为主进程，避免分布式训练中重复记录"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True
    
    def _log_message(self, message):
        """只在主进程中记录日志"""
        if self._is_main_process():
            print(message)
            logging.info(message)
            # 强制刷新日志文件，确保实时写入
            for handler in logging.getLogger().handlers:
                handler.flush()
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._is_main_process():
            current_time = time.time()
            
            msg = f"[Step {state.global_step}] "
            if "loss" in logs:
                msg += f"train_loss={logs['loss']:.4f} "
            if "learning_rate" in logs:
                msg += f"lr={logs['learning_rate']:.8f} "
            if "train_runtime" in logs:
                msg += f"runtime={logs['train_runtime']:.2f}s "
            if "train_samples_per_second" in logs:
                msg += f"samples/s={logs['train_samples_per_second']:.2f} "
            if "train_steps_per_second" in logs:
                msg += f"steps/s={logs['train_steps_per_second']:.4f} "
            
            # 添加进度信息和预估完成时间
            if self.start_time is not None:
                elapsed_time = current_time - self.start_time
                progress = state.global_step / args.max_steps
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    
                    msg += f"进度: {progress*100:.1f}% "
                    msg += f"已用时: {self._format_time(elapsed_time)} "
                    msg += f"剩余: {self._format_time(remaining_time)} "
            
            self._log_message(msg)
            self.last_log_time = current_time
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        msg = f"开始训练，总步数: {args.max_steps}"
        self._log_message(msg)
    
    def on_save(self, args, state, control, **kwargs):
        msg = f"[Step {state.global_step}] 模型已保存到 {args.output_dir}"
        self._log_message(msg)
    
config = GPT2Config(
    vocab_size=32000,         # 用自定义分词器的vocab
    n_positions=2048,         # max_token
    n_embd=768,               # 增加隐藏维度到768 (GPT2-small标准)
    n_layer=12,               # 增加层数到12
    n_head=12,                # 注意力头数调整为12 (n_embd需要被n_head整除)
    activation_function="gelu_new",
    resid_pdrop=0.1,          # 稍微增大dropout
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    use_cache=False
)

model = GPT2LMHeadModel(config)
print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 1. 加载分词器
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/byte_level_bpe_tokenizer_v1.json",
                                    bos_token="<s>", pad_token="<pad>", eos_token="</s>", unk_token="<unk>")

# GPT2架构原生没有pad token，需要手动设置
model.config.pad_token_id = tokenizer.pad_token_id 

# # 2. 加载少量数据用于快速验证（仅用于测试代码正确性）
# parquet_files = sorted(glob.glob("pretrain_data_chunks/tokenized_dataset/train_chunks_*.parquet"))[:1]  # 只取第一个文件
# full_dataset = Dataset.from_parquet(parquet_files)
# # 只取前1000条数据进行测试
# train_dataset = full_dataset.select(range(min(1000, len(full_dataset))))
# print(f"测试数据集大小: {len(train_dataset)} 条（用于验证代码）")

# 2. 加载所有parquet训练数据（预训练专注数据拟合，不使用验证集）
parquet_files = sorted(glob.glob("./pretrain_data_chunks/tokenized_dataset/train_chunks_*.parquet"))
train_dataset = Dataset.from_parquet(parquet_files)
print(f"训练数据集大小: {len(train_dataset)} 条")


# 3. 构造数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. 训练参数（预训练专注训练，最大化数据利用）
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    max_steps=200000,  # 只训练20万步,设置steps而不是epochs更适合大规模数据集
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=8,   
    save_steps=10000,
    save_total_limit=3,
    logging_steps=50,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    weight_decay=0.01,  # 添加权重衰减
    gradient_checkpointing=True,
    dataloader_pin_memory=True,  
    dataloader_num_workers=4,    
    deepspeed="ds_config.json",
    report_to=["none"],  
)

# 5. Trainer（预训练专注训练loss，无验证集）
trainer = Trainer(
    model=model,
    # processing_class=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 6. 开始训练
if __name__ == "__main__":
    trainer.add_callback(LossLoggerCallback())
    
    # 检查是否有检查点可以恢复
    checkpoints = glob.glob("./output/checkpoint-*")
    if checkpoints:
        # 找到最新的检查点
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        print(f"发现检查点: {latest_checkpoint}，从此处恢复训练...")
        logging.info(f"从检查点恢复训练: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("未发现检查点，从头开始训练...")
        logging.info("从头开始训练")
        trainer.train()
    
    trainer.save_model("./output")
    tokenizer.save_pretrained("./output")