# 奖励模型训练 - QLoRA版本
# 2025/9/10
# wenzhe

import os
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import RewardTrainer, RewardConfig
import random
import multiprocessing

num_proc = max(1, multiprocessing.cpu_count() // 2)


def build_dataset(tokenizer, dataset, max_length=1024):
    """构建RewardTrainer期望的数据格式 - 适配UltraFeedback数据集"""
    def process_ultrafeedback_batch(batch):
        instructions = batch["instruction"]
        completions_list = batch["completions"]

        batch_chosen_texts, batch_rejected_texts = [], []

        for instruction, completions in zip(instructions, completions_list):
            if len(completions) < 2:
                continue
            sorted_completions = sorted(
                completions, key=lambda x: x.get('overall_score', 0), reverse=True
            )
            chosen_response = sorted_completions[0]['response']
            rejected_response = sorted_completions[-1]['response']

            batch_chosen_texts.append(f"Instruction: {instruction}\nResponse: {chosen_response}")
            batch_rejected_texts.append(f"Instruction: {instruction}\nResponse: {rejected_response}")

        chosen_tokens = tokenizer(batch_chosen_texts, truncation=True, max_length=max_length, padding=False)
        rejected_tokens = tokenizer(batch_rejected_texts, truncation=True, max_length=max_length, padding=False)

        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }

    return dataset.map(
        process_ultrafeedback_batch,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Processing UltraFeedback dataset",
        num_proc=num_proc,
    )


class RewardModelTrainer:
    """基于QLoRA的奖励模型训练"""

    def __init__(self, base_model_path, tokenizer_path, output_dir="./reward_model", max_length=1024):
        self.base_model_path = base_model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.max_length = max_length

        os.makedirs(output_dir, exist_ok=True)
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        print(f"使用自定义分词器: {self.tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
        )

        print("使用QLoRA初始化奖励模型...")

        config = AutoConfig.from_pretrained(self.base_model_path)
        config.num_labels = 1
        config.pad_token_id = self.tokenizer.pad_token_id

        # 4-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

        # 加载预训练模型并量化
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # 预处理以适配QLoRA
        model = prepare_model_for_kbit_training(model)

        # 注入LoRA adapter
        peft_config = LoraConfig(
            r=64,
            target_modules=[
                "c_attn", "c_proj"  # GPT-2注意力层
            ],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",  # 奖励模型是分类任务
        )
        self.model = get_peft_model(model, peft_config)

        print("✅ QLoRA奖励模型初始化完成！")
        print(f"可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f}M")

    def prepare_dataset(self, dataset_name="openbmb/UltraFeedback", max_samples=None):
        print(f"加载数据集: {dataset_name}")
        ds = load_dataset(dataset_name)
        full_dataset = ds['train']

        if max_samples:
            indices = random.sample(range(len(full_dataset)), min(max_samples, len(full_dataset)))
            full_dataset = full_dataset.select(indices)

        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = build_dataset(self.tokenizer, dataset_split['train'], self.max_length)
        test_dataset = build_dataset(self.tokenizer, dataset_split['test'], self.max_length)

        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        return train_dataset, test_dataset

    def train(self, train_dataset, eval_dataset, num_train_epochs=3, learning_rate=1e-4,
              per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_steps=100,
              logging_steps=10, save_steps=500, eval_steps=500):

        training_args = RewardConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            output_dir=self.output_dir,
            fp16=False,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # 禁用wandb
            run_name="reward_model_training",  # 设置运行名称
        )

        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        print("开始训练奖励模型(QLoRA)...")
        trainer.train()
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(self.output_dir)
        return trainer


def main():
    reward_trainer = RewardModelTrainer(
        base_model_path="../output",
        tokenizer_path="../tokenizer/byte_level_bpe_tokenizer_v1.json",
        output_dir="./reward_model_output",
        max_length=1024,
    )

    train_dataset, eval_dataset = reward_trainer.prepare_dataset(
        dataset_name="openbmb/UltraFeedback", max_samples=5000
    )

    reward_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=2,
        learning_rate=1e-4,  # QLoRA常用稍大的lr
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
    )

    print("奖励模型(QLoRA)训练完成！")


if __name__ == "__main__":
    main()
