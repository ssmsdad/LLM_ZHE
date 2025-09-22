#!/usr/bin/env python3
"""
加载QLoRA微调后的模型进行推理
支持两种加载方式：
1. 加载LoRA适配器 + 基础模型
2. 直接加载合并后的完整模型
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedTokenizerFast,
    BitsAndBytesConfig
)
from peft import PeftModel
import argparse


class ModelLoader:
    def __init__(self, tokenizer_path="../tokenizer/byte_level_bpe_tokenizer_v1.json"):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.model = None
        
    def load_tokenizer(self):
        """加载分词器"""
        print(f"加载分词器: {self.tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", 
            pad_token="<pad>", 
            eos_token="</s>", 
            unk_token="<unk>",
        )
        return self.tokenizer
    
    def load_merged_model(self, merged_model_path="./sft_output_merged"):
        """
        方法1: 加载合并后的完整模型 (推荐)
        优点: 简单直接，无需量化配置
        缺点: 模型文件较大
        """
        print(f"加载合并后的完整模型: {merged_model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配设备
            trust_remote_code=True,
        )
        
        if self.tokenizer is None:
            self.load_tokenizer()
            
        print("✅ 合并模型加载完成")
        return self.model
    
    def load_lora_model(self, 
                       base_model_path="../output", 
                       lora_adapter_path="./sft_output",
                       use_4bit=True):
        """
        方法2: 加载LoRA适配器 + 基础模型
        优点: 节省存储空间，可以切换不同的LoRA适配器
        缺点: 需要基础模型和量化配置
        """
        print(f"加载基础模型: {base_model_path}")
        print(f"加载LoRA适配器: {lora_adapter_path}")
        
        if use_4bit:
            # 4bit量化配置 (与训练时保持一致)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # 加载量化的基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            # 不使用量化
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            base_model, 
            lora_adapter_path,
            torch_dtype=torch.float16,
        )
        
        if self.tokenizer is None:
            self.load_tokenizer()
            
        print("✅ LoRA模型加载完成")
        return self.model
    
    def generate_text(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """生成文本 - 改进版，避免重复输出"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先加载模型和分词器")
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        # 改进的生成参数，避免重复
        generation_config = {
            "max_new_tokens": 256,  # 🔧 使用max_new_tokens而不是max_length
            "min_length": len(inputs[0]) + 10,  # 🔧 设置最小长度
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,  # 🔧 添加top_k限制
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # 🔧 添加重复惩罚
            "no_repeat_ngram_size": 3,  # 🔧 避免3-gram重复
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_config)
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 去除输入部分，只返回生成的部分
        generated_only = generated_text[len(prompt):].strip()
        return generated_only


def main():
    parser = argparse.ArgumentParser(description="加载QLoRA微调模型进行推理")
    parser.add_argument("--mode", choices=["merged", "lora"], default="merged",
                       help="加载模式: merged(合并模型) 或 lora(LoRA适配器)")
    parser.add_argument("--merged_path", default="./sft_output_merged",
                       help="合并模型路径")
    parser.add_argument("--base_model", default="../output",
                       help="基础模型路径")
    parser.add_argument("--lora_path", default="./sft_output",
                       help="LoRA适配器路径")
    parser.add_argument("--no_4bit", action="store_true",
                       help="不使用4bit量化 (仅对LoRA模式有效)")
    
    args = parser.parse_args()
    
    # 初始化加载器
    loader = ModelLoader()
    
    try:
        if args.mode == "merged":
            # 方法1: 加载合并后的完整模型
            loader.load_merged_model(args.merged_path)
            
        elif args.mode == "lora":
            # 方法2: 加载LoRA适配器
            use_4bit = not args.no_4bit
            loader.load_lora_model(
                base_model_path=args.base_model,
                lora_adapter_path=args.lora_path,
                use_4bit=use_4bit
            )
        
        # 交互式对话
        print("\n" + "="*50)
        print("🤖 模型加载完成！开始对话 (输入 'quit' 退出)")
        print("="*50)
        
        while True:
            user_input = input("\n👤 User: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # 格式化为训练时的格式
            prompt = f"User: {user_input}\nAssistant:"
            
            # 生成回复
            print("🤖 Assistant: ", end="", flush=True)
            response = loader.generate_text(
                prompt, 
                max_length=512, 
                temperature=0.7, 
                top_p=0.9
            )
            print(response)
            
    except KeyboardInterrupt:
        print("\n👋 对话结束")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
