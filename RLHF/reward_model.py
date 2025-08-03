# 奖励模型训练 - 基于人类反馈的强化学习第一步
# 2025/7/29

# 使用TRL库的RewardTrainer进行奖励模型训练
# RewardModelTrainer 与 transformers的Trainer相比，有以下特点：
# 1. 支持奖励模型的特殊数据格式（chosen和rejected）
# 2. loss是pairwise的，适用于强化学习中的奖励模型训练

import os
# 设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2ForSequenceClassification
)
from trl import RewardTrainer
import random
import multiprocessing

num_proc = max(1, multiprocessing.cpu_count() // 2)

def build_dataset(tokenizer, dataset, max_length=1024):
    """
    构建RewardTrainer期望的数据格式 - 适配UltraFeedback数据集
    使用datasets的map函数进行批处理，提高效率
    """
    def process_ultrafeedback_batch(batch):
        """批处理UltraFeedback样本"""
        instructions = batch["instruction"]
        completions_list = batch["completions"]
        
        batch_chosen_texts = []
        batch_rejected_texts = []
        
        for instruction, completions in zip(instructions, completions_list):
            # 跳过回答数量不足的样本
            if len(completions) < 2:
                continue
                
            # 根据overall_score排序，选择最高分和最低分
            sorted_completions = sorted(completions, key=lambda x: x.get('overall_score', 0), reverse=True)
            
            # 构造chosen和rejected文本
            chosen_response = sorted_completions[0]['response']
            rejected_response = sorted_completions[-1]['response']
            
            # 组合instruction和response
            chosen_text = f"Instruction: {instruction}\nResponse: {chosen_response}"
            rejected_text = f"Instruction: {instruction}\nResponse: {rejected_response}"
            
            batch_chosen_texts.append(chosen_text)
            batch_rejected_texts.append(rejected_text)
        
        chosen_tokens = tokenizer(
            batch_chosen_texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        rejected_tokens = tokenizer(
            batch_rejected_texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }
    
    processed_dataset = dataset.map(
        process_ultrafeedback_batch,
        batched=True,
        batch_size=1000,  # 每批处理1000个样本
        remove_columns=dataset.column_names,  # 删除原始列
        desc="Processing UltraFeedback dataset",  # 显示进度条
        num_proc=num_proc,  # 使用多进程
    )
    
    return processed_dataset

class RewardModelTrainer:
    """奖励模型训练主类"""
    
    def __init__(
        self,
        base_model_name: str = None,  # 奖励模型不需要预训练模型
        tokenizer_path: str = None,
        output_dir: str = "./reward_model",
        max_length: int = 1024
    ):
        self.base_model_name = base_model_name
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.max_length = max_length
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.setup_model_and_tokenizer()
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        
        # 加载分词器
        print(f"使用自定义分词器: {self.tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
        
        # 奖励模型得我架构要与预训练模型相同
        print("初始化奖励模型（随机权重）...")    
        config = GPT2Config(
            vocab_size=32000,
            n_positions=2048,
            n_embd=768,
            n_layer=12,
            n_head=12,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            num_labels=1,  # 输出一个奖励分数
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 模型最终的输出为[batch, num_labels]，表示每个样本的奖励分数
        self.model = GPT2ForSequenceClassification(config)
        
        print("奖励模型初始化完成！（使用随机权重）")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
    
    def prepare_dataset(self, dataset_name="openbmb/UltraFeedback", max_samples=None):
        """准备训练数据集 - 使用UltraFeedback数据集"""
        print(f"加载数据集: {dataset_name}")
        
        ds = load_dataset(dataset_name)
        full_dataset = ds['train']
        
        # 限制样本数量
        if max_samples:
            indices = random.sample(range(len(full_dataset)), min(max_samples, len(full_dataset)))
            full_dataset = full_dataset.select(indices)
        
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        test_dataset = dataset_split['test']
        
        # 预处理为RewardTrainer期望的格式
        train_dataset = build_dataset(self.tokenizer, train_dataset, self.max_length)
        test_dataset = build_dataset(self.tokenizer, test_dataset, self.max_length)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-5,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
    ):
        """开始训练奖励模型"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=True, 
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("开始训练奖励模型...")
        trainer.train()

        print("保存奖励模型...")
        trainer.save_model()
        # self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    """主函数"""
    print("开始训练奖励模型...")
    
    # 配置 - 奖励模型不需要base_model_name，使用随机初始化
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"  # 自定义分词器
    
    # 初始化训练器
    reward_trainer = RewardModelTrainer(
        base_model_name=None,  # 不需要预训练模型
        tokenizer_path=tokenizer_path,
        output_dir="./reward_model",
        max_length=1024
    )
    
    # 准备数据集
    train_dataset, eval_dataset = reward_trainer.prepare_dataset(
        dataset_name="openbmb/UltraFeedback",
        max_samples=5000  # UltraFeedback数据集较大，限制样本数量
    )
    
    # 开始训练
    reward_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=2,  # 根据显存调整
        gradient_accumulation_steps=8,  # 有效batch size = 2 * 8 = 16
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
    )
    
    print("奖励模型训练完成！")
    print(f"模型已保存到: {reward_trainer.output_dir}")

if __name__ == "__main__":
    main()