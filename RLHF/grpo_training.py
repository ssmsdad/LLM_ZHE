# GRPO (Group Relative Policy Optimization) Training
# 2025/8/2
# wenzhe

# GRPO是对PPO的改进算法，特别适用于RLHF场景
# 主要优势：
# 1. 更稳定的训练过程
# 2. 更好的样本效率
# 3. 相对比较容易收敛

import os
# 设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import random
from typing import List, Dict
import numpy as np

class GRPOTrainer:
    """GRPO训练器 - 基于TRL的PPO实现GRPO算法"""
    
    def __init__(
        self,
        sft_model_path: str,
        reward_model_path: str,
        tokenizer_path: str = "../tokenizer/byte_level_bpe_tokenizer_v1.json",
        output_dir: str = "./grpo_model",
        max_length: int = 1024,
    ):
        self.sft_model_path = sft_model_path
        self.reward_model_path = reward_model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.max_length = max_length
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.setup_models_and_tokenizer()
    
    def setup_models_and_tokenizer(self):
        """设置策略模型、奖励模型和分词器"""
        print("Loading models and tokenizer...")
        
        # 加载分词器
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", 
            pad_token="<pad>", 
            eos_token="</s>", 
            unk_token="<unk>"
        )
        
        # 配置4bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载策略模型（SFT模型）- 带价值头用于PPO
        print(f"Loading policy model: {self.sft_model_path}")
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.sft_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 加载奖励模型
        print(f"Loading reward model: {self.reward_model_path}")
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            self.reward_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 确保pad_token_id设置正确
        if self.policy_model.config.pad_token_id is None:
            self.policy_model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.reward_model.config.pad_token_id is None:
            self.reward_model.config.pad_token_id = self.tokenizer.pad_token_id

        # 为量化模型准备训练环境
        self.policy_model = prepare_model_for_kbit_training(self.policy_model)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                # 对于带价值头的模型，可能需要包含价值头的参数
                # "v_head",  # 取消注释如果需要微调价值头
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 将LoRA适配器"插入"到模型中
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        self.policy_model.print_trainable_parameters()
        
        print("Models loaded successfully!")
    
    def prepare_dataset(self, dataset_name="vicgalle/alpaca-gpt4", max_samples=1000):
        """准备GRPO训练数据集 - 适配alpaca-gpt4格式"""
        print(f"Loading dataset: {dataset_name}")
        
        # 加载数据集
        ds = load_dataset(dataset_name)
        train_dataset = ds['train']
        
        # 限制样本数量
        if max_samples:
            indices = random.sample(range(len(train_dataset)), min(max_samples, len(train_dataset)))
            train_dataset = train_dataset.select(indices)
        
        def process_alpaca_batch(batch):
            """批处理alpaca-gpt4数据集，提取instruction作为query"""
            queries = []
            
            for i in range(len(batch['instruction'])):
                instruction = batch['instruction'][i]
                input_text = batch['input'][i] if batch['input'][i] else ""
                
                # 构造query：instruction + input（如果有的话）
                if input_text:
                    query = f"{instruction}\n{input_text}"
                else:
                    query = instruction
                
                queries.append(query.strip())
            
            return {"query": queries}
        
        # 使用map批处理提取queries
        processed_dataset = train_dataset.map(
            process_alpaca_batch,
            batched=True,
            batch_size=1000,
            remove_columns=train_dataset.column_names,
            desc="Processing Alpaca-GPT4 dataset"
        )
        
        # 提取查询列表
        queries = processed_dataset['query']
        
        print(f"Prepared {len(queries)} training queries")
        return queries[:max_samples] if max_samples else queries
    
    def generate_responses(self, queries: List[str], batch_size: int = 4):
        """使用策略模型生成回应"""
        responses = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            
            # 编码queries
            encoded_queries = [
                self.tokenizer.encode(query, return_tensors="pt").squeeze()
                for query in batch_queries
            ]
            
            # 生成回应
            with torch.no_grad():
                for query_tensor in encoded_queries:
                    query_tensor = query_tensor.to(self.policy_model.device)
                    
                    # 生成回应
                    response_tensor = self.policy_model.generate(
                        query_tensor.unsqueeze(0),
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # 提取新生成的部分
                    response_only = response_tensor[0][len(query_tensor):]
                    response_text = self.tokenizer.decode(response_only, skip_special_tokens=True)
                    responses.append(response_text)
        
        return responses
    
    def compute_rewards(self, queries: List[str], responses: List[str]) -> List[float]:
        """使用奖励模型计算奖励分数"""
        rewards = []
        
        for query, response in zip(queries, responses):
            # 组合query和response
            full_text = f"{query}\n{response}"
            
            # 编码
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
            
            # 计算奖励
            with torch.no_grad():
                reward_output = self.reward_model(**inputs)
                # 奖励模型输出logits，取第一个值作为奖励分数
                reward_score = reward_output.logits[0, 0].item()
                rewards.append(reward_score)
        
        return rewards
    
    def train(
        self,
        queries: List[str],
        learning_rate: float = 1e-5,
        mini_batch_size: int = 4,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 4,
        ppo_epochs: int = 4,
        max_grad_norm: float = 1.0,
        cliprange: float = 0.2,
        vf_coef: float = 0.1,
        target_kl: float = 0.01,
        num_train_epochs: int = 3,
    ):
        """开始GRPO训练"""
        
        # 配置PPO参数（GRPO基于PPO）
        ppo_config = PPOConfig(
            model_name=self.sft_model_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ppo_epochs=ppo_epochs,
            max_grad_norm=max_grad_norm,
            cliprange=cliprange,
            vf_coef=vf_coef,
            target_kl=target_kl,
            remove_unused_columns=False,
        )
        
        # 创建PPO训练器
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            tokenizer=self.tokenizer,
        )
        
        print("Starting GRPO training...")
        
        for epoch in range(num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{num_train_epochs}")
            
            # 随机采样queries用于这个epoch
            epoch_queries = random.sample(queries, min(batch_size, len(queries)))
            
            # 生成回应
            print("Generating responses...")
            responses = self.generate_responses(epoch_queries, mini_batch_size)
            
            # 计算奖励
            print("Computing rewards...")
            rewards = self.compute_rewards(epoch_queries, responses)
            
            # 转换为tensors
            query_tensors = [
                self.tokenizer.encode(q, return_tensors="pt").squeeze()
                for q in epoch_queries
            ]
            response_tensors = [
                self.tokenizer.encode(r, return_tensors="pt").squeeze()
                for r in responses
            ]
            reward_tensors = [torch.tensor(r, dtype=torch.float) for r in rewards]
            
            # PPO训练步骤
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            # 打印统计信息
            print(f"Reward mean: {np.mean(rewards):.4f}")
            print(f"Reward std: {np.std(rewards):.4f}")
            if stats:
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.4f}")
        
        # 保存最终模型
        print("Saving GRPO model...")
        self.policy_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return ppo_trainer

def main():
    """主函数"""
    print("Starting GRPO training...")
    
    # 配置路径
    sft_model_path = "../SFT/qlora_chatbot_model_trl"  # SFT模型路径
    reward_model_path = "./reward_model"  # 奖励模型路径
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"
    
    # 初始化GRPO训练器
    grpo_trainer = GRPOTrainer(
        sft_model_path=sft_model_path,
        reward_model_path=reward_model_path,
        tokenizer_path=tokenizer_path,
        output_dir="./grpo_model",
        max_length=1024,
    )
    
    # 准备数据集
    queries = grpo_trainer.prepare_dataset(
        dataset_name="vicgalle/alpaca-gpt4",
        max_samples=500  # 小规模测试，实际可以增加
    )
    
    # 开始训练
    grpo_trainer.train(
        queries=queries,
        learning_rate=1e-6,  # GRPO使用较小的学习率
        mini_batch_size=2,   # 根据显存调整
        batch_size=8,        # 每个epoch的样本数
        gradient_accumulation_steps=4,
        ppo_epochs=4,        # 每批数据的PPO轮数
        num_train_epochs=2,  # 总训练轮数
        target_kl=0.01,     # KL散度约束
    )
    
    print("GRPO training completed!")
    print(f"Model saved to: {grpo_trainer.output_dir}")
    print("\nNext steps:")
    print("1. 测试GRPO微调后的模型性能")
    print("2. 与原始SFT模型进行对比评估")
    print("3. 根据需要调整超参数继续训练")

if __name__ == "__main__":
    main()
