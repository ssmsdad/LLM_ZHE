# GRPO Training using TRL's GRPOTrainer
# 2025/8/2
# wenzhe

# 使用TRL库的专用GRPOTrainer进行强化学习微调
# GRPOTrainer相比PPOTrainer的优势：
# 1. 专门为GRPO算法设计，更稳定
# 2. 自动处理组相对优化逻辑
# 3. 更好的内存管理和训练效率

import os
# 设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免多进程tokenizer警告
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    AutoModelForSequenceClassification
)
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel
import random
from typing import List
import multiprocessing

num_proc = max(1, multiprocessing.cpu_count() // 2)

class GRPOTrainerTRL:
    """基于TRL GRPOTrainer的强化学习微调"""
    
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        reward_model_path: str,
        tokenizer_path: str = "../tokenizer/byte_level_bpe_tokenizer_v1.json",
        output_dir: str = "./grpo_model_trl",
        max_length: int = 1024,
    ):
        self.base_model_path = base_model_path
        self.sft_model_path = adapter_path
        self.reward_model_path = reward_model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.max_length = max_length
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.setup_model_and_tokenizer()
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        print("Loading model and tokenizer...")
        
        # 加载分词器
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", 
            pad_token="<pad>", 
            eos_token="</s>", 
            unk_token="<unk>",
        )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))   # 每个进程只在自己的 GPU 上放一份完整模型
        
        # 配置4bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载策略模型（SFT模型）
        print(f"Loading policy model: {self.base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map={"": torch.device(f"cuda:{local_rank}")},   # 固定到当前进程的卡
            # device_map={"": local_rank},   # <---- 关键
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 确保pad_token_id设置正确
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 加载SFT适配器到基础模型上
        self.ft_model = PeftModel.from_pretrained(self.model, self.sft_model_path)
        
        # 确保LoRA参数可训练（重要！）
        self.ft_model.train()
        for name, param in self.ft_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True


        print(f"Loading reward model: {self.reward_model_path}")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.device(f"cuda:{local_rank}")},   # 固定到当前进程的卡
            # device_map={"": local_rank},   # <---- 关键
            trust_remote_code=True,
        )
        
        print("All Model and tokenizer loaded successfully!")
        trainable = sum(p.numel() for p in self.ft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.ft_model.parameters())
        print(f"Trainable parameters: {trainable}/{total} ({100*trainable/total:.2f}%)")
    
    def prepare_dataset(self, dataset_name="vicgalle/alpaca-gpt4",max_samples=None):
        """准备GRPO训练数据集 - 适配alpaca-gpt4格式"""
        print(f"Loading dataset: {dataset_name}")
        
        # 加载数据集
        ds = load_dataset(dataset_name)
        train_dataset = ds['train']
        
        # 限制样本数量进行快速测试
        if max_samples:
            indices = random.sample(range(len(train_dataset)), min(max_samples, len(train_dataset)))
            train_dataset = train_dataset.select(indices)
        
        def process_alpaca_batch(batch):
            """批处理alpaca-gpt4数据集，提取instruction作为prompt"""
            prompts = []
            
            for i in range(len(batch['instruction'])):
                instruction = batch['instruction'][i]
                input_text = batch['input'][i] if batch['input'][i] else ""
                
                # 构造prompt：instruction + input（如果有的话）
                if input_text:
                    prompt = f"{instruction}\n{input_text}"
                else:
                    prompt = instruction
                
                # 严格限制prompt长度，防止形状错误
                # 先用字符长度粗略过滤
                if len(prompt) > 1500:  # 字符长度限制
                    prompt = prompt[:1500]
                
                # 再用分词器精确截断
                try:
                    prompt_tokens = self.tokenizer(
                        prompt, 
                        max_length=350,  # 更严格的token限制
                        truncation=True,
                        add_special_tokens=False
                    )['input_ids']
                    prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                except Exception as e:
                    # 如果分词出错，使用字符截断
                    print(f"Tokenization error, using character truncation: {e}")
                    prompt = prompt[:500]
                
                prompts.append(prompt.strip())
            
            return {"prompt": prompts}
        
        # 使用map批处理，禁用多进程避免CUDA冲突
        processed_dataset = train_dataset.map(
            process_alpaca_batch,
            batched=True,
            batch_size=1000,  # 每批处理1000个样本
            remove_columns=train_dataset.column_names,  # 删除原始列
            desc="Processing Alpaca-GPT4 dataset",  # 显示进度条
            num_proc=1,  # 禁用多进程，避免CUDA重初始化错误
        )
        
        print(f"Prepared {len(processed_dataset)} training samples")
        return processed_dataset
    
    def create_reward_function(self):
        """创建奖励函数"""
        
        def reward_function(*args, **kwargs) -> List[float]:
            """
            奖励函数 - 兼容TRL GRPOTrainer的多种调用方式
            支持：
            - reward_function(prompts, responses)
            - reward_function(prompts=..., responses=..., completions=...)
            - 其他TRL可能的调用方式
            """
            # 解析参数
            if len(args) >= 2:
                # 位置参数调用: reward_function(prompts, responses)
                prompts, responses = args[0], args[1]
            elif 'prompts' in kwargs and 'responses' in kwargs:
                # 关键字参数调用: reward_function(prompts=..., responses=...)
                prompts, responses = kwargs['prompts'], kwargs['responses']
            elif 'prompts' in kwargs and 'completions' in kwargs:
                # 新版TRL可能的调用方式
                prompts, responses = kwargs['prompts'], kwargs['completions']
            else:
                raise ValueError(f"Invalid reward function call: args={args}, kwargs={kwargs}")
            
            rewards = []
            
            for prompt, response in zip(prompts, responses):
                # 组合prompt和response，构造完整的对话
                full_text = f"Instruction: {prompt}\nResponse: {response}"
                
                # 分词
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
                
                # 计算奖励分数
                with torch.no_grad():
                    outputs = self.reward_model(**inputs)
                    # 奖励模型输出单个分数 [batch_size, 1]
                    reward_score = outputs.logits.squeeze().item()
                    rewards.append(reward_score)
            
            return rewards
        
        return reward_function
    
    def train(
        self,
        train_dataset,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-6,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
    ):
        """开始GRPO训练"""
        
        # 创建奖励函数
        reward_fn = self.create_reward_function()
        
        # 配置GRPO训练参数
        grpo_config = GRPOConfig(
            learning_rate=learning_rate,
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            report_to="none",
            # GRPO特有参数
            beta=0.3,  # GRPO的重要参数，控制policy和reference model的差异
            # scale_rewards=True,  # 是否缩放奖励
            # 防止形状错误的配置
            remove_unused_columns=False,
            dataloader_drop_last=True,
            ignore_data_skip=True,  # 跳过有问题的数据
            # 明确指定从最新检查点恢复
            resume_from_checkpoint="./grpo_model_output/checkpoint-4000",
        )

        # 创建GRPO训练器（新版接口）
        trainer = GRPOTrainer(
            model=self.ft_model,
            reward_funcs=reward_fn,  # 这里是函数或函数列表
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,  # 主tokenizer
            reward_processing_classes=self.tokenizer,  # 奖励模型tokenizer
        )

        
        # 开始训练
        print("Starting GRPO training with TRL GRPOTrainer...")
        # trainer.train(resume_from_checkpoint="./grpo_model_output/checkpoint-7000")
        trainer.train()
        
        # 保存最终模型
        print("Saving GRPO model...")
        trainer.save_model()
        # self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    """主函数"""
    # 配置路径
    base_model_path = "../output"  # 基础模型路径
    adapter_path = "../SFT/sft_output"  # SFT模型路径
    reward_model_path = "./reward_model_output"  # 奖励模型路径
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"
    
    # 初始化GRPO训练器
    grpo_trainer = GRPOTrainerTRL(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        reward_model_path=reward_model_path,
        tokenizer_path=tokenizer_path,
        output_dir="./grpo_model_output_new",
        max_length=1024,
    )
    # 准备数据集
    train_dataset = grpo_trainer.prepare_dataset(
        dataset_name="vicgalle/alpaca-gpt4",
        # max_samples=20000  # 小规模测试
        max_samples=2000  # 小规模测试
    )
    
    # 开始训练
    grpo_trainer.train(
        train_dataset=train_dataset,
        num_train_epochs=2,
        learning_rate=1e-6,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=8,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        logging_steps=50,
        save_steps=500,
    )
    
    print("GRPO training completed!")
    print(f"Model saved to: {grpo_trainer.output_dir}")

if __name__ == "__main__":
    main()
