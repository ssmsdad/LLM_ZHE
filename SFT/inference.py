# 推理脚本 - 测试微调后的QLoRA模型
# 2025/7/28
# wenzhe

import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class ChatbotInference:
    """微调后聊天机器人的推理类"""
    
    def __init__(self, base_model_name: str, peft_model_path: str, tokenizer_path: str = None):
        """
        初始化推理模型
        
        Args:
            base_model_name: 基础模型名称
            peft_model_path: LoRA适配器路径
            tokenizer_path: 自定义分词器路径（可选）
        """
        print(f"加载基础模型: {base_model_name}")
        
        # 配置4bit量化（与训练时保持一致）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载分词器
        if tokenizer_path:
            # 使用自定义分词器
            print(f"加载自定义分词器: {tokenizer_path}")
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                bos_token="<s>", 
                pad_token="<pad>", 
                eos_token="</s>", 
                unk_token="<unk>"
            )
        else:
            # 使用模型默认分词器
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
        
        # 加载基础模型（使用4bit量化）
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,  # 添加量化配置
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 加载LoRA适配器
        print(f"加载LoRA适配器: {peft_model_path}")
        self.model = PeftModel.from_pretrained(self.model, peft_model_path)
        
        # 合并适配器权重（可选，用于更快的推理）
        # self.model = self.model.merge_and_unload()
        
        self.model.eval()
        print("模型加载完成！")
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        生成回复
        
        Args:
            user_input: 用户输入
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样，保证输出结果多样性
            
        Returns:
            生成的回复
        """
        # 格式化输入
        prompt = f"User: {user_input}\nAssistant:"
        
        # 编码输入，self.tokenizer.encode只返回input_ids（二维list），没有attention_mask
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,  # 使用专门的pad token
                eos_token_id=self.tokenizer.eos_token_id,  # 使用EOS token结束生成
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手的回复
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat_loop(self):
        """交互式聊天循环"""
        print("开始对话！输入 'quit' 退出。")
        print("-" * 50)
        
        while True:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            try:
                response = self.generate_response(user_input)
                print(f"Assistant: {response}")
                print("-" * 50)
            except Exception as e:
                print(f"生成回复时出错: {e}")

def main():
    """主函数"""
    # 配置模型路径
    base_model_name = "microsoft/DialoGPT-small"  # 替换为你的基础模型
    peft_model_path = "./qlora_chatbot_model"  # LoRA适配器路径
    tokenizer_path = "../tokenizer/byte_level_bpe_tokenizer_v1.json"  # 自定义分词器路径
    
    try:
        # 初始化推理模型
        chatbot = ChatbotInference(
            base_model_name=base_model_name,
            peft_model_path=peft_model_path,
            tokenizer_path=tokenizer_path  # 使用自定义分词器
        )
        
        # 测试几个问题
        test_questions = [
            "Hello! How are you?",
            "What is the difference between Python and JavaScript?",
            "Can you help me with machine learning?",
            "What's the weather like today?",
        ]
        
        print("测试模型回复:")
        print("=" * 60)
        
        for question in test_questions:
            print(f"User: {question}")
            response = chatbot.generate_response(question)
            print(f"Assistant: {response}")
            print("-" * 50)
        
        # 启动交互式聊天
        print("\n切换到交互模式:")
        chatbot.chat_loop()
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("请确保模型路径正确且训练已完成。")

if __name__ == "__main__":
    main()
