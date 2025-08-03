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
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    AutoModelForSequenceClassification
)
from trl import GRPOTrainer, GRPOConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import random
from typing import Dict, List
import multiprocessing

num_proc = max(1, multiprocessing.cpu_count() // 2)

class GRPOTrainerTRL:
    """基于TRL GRPOTrainer的强化学习微调"""
    
    def __init__(
        self,
        sft_model_path: str,
        reward_model_path: str,
        tokenizer_path: str = "../tokenizer/byte_level_bpe_tokenizer_v1.json",
        output_dir: str = "./grpo_model_trl",
        max_length: int = 1024,
    ):
        self.sft_model_path = sft_model_path
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
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
        
        # 配置4bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载策略模型（SFT模型）
        print(f"Loading policy model: {self.sft_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 确保pad_token_id设置正确
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 添加LoRA适配器
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print(f"Loading reward model: {self.reward_model_path}")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("All Model and tokenizer loaded successfully!")
    
    def prepare_dataset(self, dataset_name="vicgalle/alpaca-gpt4", max_samples=1000):
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
                
                prompts.append(prompt.strip())
            
            return {"prompt": prompts}
        
        # 使用map批处理，大幅提升速度
        processed_dataset = train_dataset.map(
            process_alpaca_batch,
            batched=True,
            batch_size=1000,  # 每批处理1000个样本
            remove_columns=train_dataset.column_names,  # 删除原始列
            desc="Processing Alpaca-GPT4 dataset",  # 显示进度条
            num_proc=num_proc,  # 使用多进程加速
        )
        
        print(f"Prepared {len(processed_dataset)} training samples")
        return processed_dataset
    
    def create_reward_function(self):
        """创建奖励函数"""
        
        def reward_function(prompts: List[str], responses: List[str]) -> List[float]:
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
                    reward_score = outputs.logits[0, 0].item()
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
        max_new_tokens: int = 150,
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
            batch_size=per_device_train_batch_size * gradient_accumulation_steps,
            mini_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            # GRPO特定参数
            rpo_alpha=1.0,  # 相对策略优化的权重
            use_score_scaling=True,  # 使用分数缩放
            score_clip=5.0,  # 分数裁剪范围
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            bf16=True,
            remove_unused_columns=False,
            report_to="none",  # 可以改为"wandb"如果需要
        )
        
        # 创建GRPO训练器
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            grpo_config=grpo_config,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            reward_function=reward_fn,
        )
        
        # 开始训练
        print("Starting GRPO training with TRL GRPOTrainer...")
        trainer.train()
        
        # 保存最终模型
        print("Saving GRPO model...")
        trainer.save_model()
        # self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    """主函数"""
    print("Starting GRPO training with TRL GRPOTrainer...")
    
    # 配置路径
    sft_model_path = "../SFT/qlora_chatbot_model_trl"  # SFT模型路径
    reward_model_path = "./reward_model"  # 奖励模型路径
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"
    
    # 初始化GRPO训练器
    grpo_trainer = GRPOTrainerTRL(
        sft_model_path=sft_model_path,
        reward_model_path=reward_model_path,
        tokenizer_path=tokenizer_path,
        output_dir="./grpo_model_trl",
        max_length=1024,
    )
    
    # 准备数据集
    train_dataset = grpo_trainer.prepare_dataset(
        dataset_name="vicgalle/alpaca-gpt4",
        max_samples=200  # 小规模测试
    )
    
    # 开始训练
    grpo_trainer.train(
        train_dataset=train_dataset,
        num_train_epochs=2,
        learning_rate=1e-6,
        per_device_train_batch_size=1,  # 根据显存调整
        gradient_accumulation_steps=8,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        logging_steps=5,
        save_steps=100,
    )
    
    print("GRPO training completed!")
    print(f"Model saved to: {grpo_trainer.output_dir}")
    
    print("Next steps:")
    print("- Test the final GRPO model performance")
    print("- Compare with SFT model to see improvements")

if __name__ == "__main__":
    main()
