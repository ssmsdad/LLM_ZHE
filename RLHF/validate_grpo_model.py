#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLHF模型验证脚本
验证经过GRPO训练后的模型能力
"""

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel
import argparse

class RLHFModelValidator:
    """RLHF模型验证器"""
    
    def __init__(
        self,
        base_model_path: str = "../output",
        grpo_adapter_path: str = "./grpo_model_output", 
        tokenizer_path: str = "../tokenizer/byte_level_bpe_tokenizer_v1.json"
    ):
        self.base_model_path = base_model_path
        self.grpo_adapter_path = grpo_adapter_path
        self.tokenizer_path = tokenizer_path
        
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("Loading RLHF model...")
        
        # 加载分词器
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token="<s>", 
            pad_token="<pad>", 
            eos_token="</s>", 
            unk_token="<unk>",
        )
        
        # 加载基础模型
        print(f"Loading base model: {self.base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 加载GRPO训练后的适配器
        print(f"Loading GRPO adapter: {self.grpo_adapter_path}")
        self.rlhf_model = PeftModel.from_pretrained(
            self.base_model, 
            self.grpo_adapter_path
        )
        
        # 设置pad_token_id
        if self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
            self.rlhf_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("RLHF model loaded successfully!")
        
        # 显示模型参数信息
        total_params = sum(p.numel() for p in self.rlhf_model.parameters())
        print(f"Total model parameters: {total_params:,}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """生成回复"""
        
        # 格式化输入
        formatted_input = f"Instruction: {prompt}\nResponse:"
        
        # 分词
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(self.rlhf_model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.rlhf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复部分
        if "Response:" in full_response:
            response = full_response.split("Response:")[-1].strip()
        else:
            response = full_response.strip()
        
        return response
    
    def compare_models(self, prompt: str):
        """对比基础模型和RLHF模型的输出"""
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        # 基础模型输出
        print("\n🤖 Base Model Response:")
        print("-" * 40)
        formatted_input = f"Instruction: {prompt}\nResponse:"
        inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            base_outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty = 1.1,
                no_repeat_ngram_size=3,
            )
        
        base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        if "Response:" in base_response:
            base_response = base_response.split("Response:")[-1].strip()
        print(base_response)
        
        # RLHF模型输出
        print("\n✨ RLHF Model Response:")
        print("-" * 40)
        rlhf_response = self.generate_response(prompt)
        print(rlhf_response)
        
        print(f"\n{'='*60}\n")
    
    def interactive_test(self):
        """交互式测试"""
        print("\n🚀 RLHF Model Interactive Test")
        print("Type 'quit' to exit, 'compare' before your prompt to compare both models")
        
        while True:
            try:
                user_input = input("\nEnter your instruction: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower().startswith('compare'):
                    prompt = user_input[7:].strip()  # 移除'compare'
                    if prompt:
                        self.compare_models(prompt)
                    else:
                        print("Please provide a prompt after 'compare'")
                elif user_input:
                    response = self.generate_response(user_input)
                    print(f"\n✨ RLHF Response: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run_predefined_tests(self):
        """运行预定义测试用例"""
        test_cases = [
            "如何学习机器学习？",
            "请解释什么是深度学习",
            "Write a Python function to calculate factorial",
            "What are the benefits of renewable energy?",
            "How to cook pasta properly?",
            "Explain the theory of relativity in simple terms",
            "给我一些减压的建议",
            "What is the difference between AI and ML?"
        ]
        
        print("🧪 Running Predefined Test Cases")
        print("=" * 50)
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n[Test {i}] {prompt}")
            print("-" * 40)
            response = self.generate_response(prompt)
            print(response)
            print()

def main():
    parser = argparse.ArgumentParser(description="Validate RLHF Model")
    parser.add_argument("--base_model", type=str, default="../output", help="Base model path")
    # parser.add_argument("--adapter", type=str, default="./grpo_model_output/checkpoint-500", help="GRPO adapter path") 
    parser.add_argument("--adapter", type=str, default="./grpo_model_output_new/", help="GRPO adapter path") 
    parser.add_argument("--tokenizer", type=str, default="../tokenizer/byte_level_bpe_tokenizer_v1.json", help="Tokenizer path")
    parser.add_argument("--mode", type=str, choices=["test", "interactive", "compare"], default="interactive", help="Test mode")
    
    args = parser.parse_args()
    
    # 初始化验证器
    validator = RLHFModelValidator(
        base_model_path=args.base_model,
        grpo_adapter_path=args.adapter,
        tokenizer_path=args.tokenizer
    )
    
    if args.mode == "test":
        validator.run_predefined_tests()
    elif args.mode == "interactive":
        validator.interactive_test()
    elif args.mode == "compare":
        # 比较模式示例
        test_prompts = [
            "如何学习编程？",
            "Please explain artificial intelligence",
            "给我一些健康饮食的建议"
        ]
        for prompt in test_prompts:
            validator.compare_models(prompt)

if __name__ == "__main__":
    main()
