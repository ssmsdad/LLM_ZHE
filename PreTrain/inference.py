
# 验证预训练模型的推理能力
# 2025/8/3
# wenzhe

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer
import time

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    print(f"Loading model: {model_path}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用半精度加速推理
        device_map="auto",  # 自动分配设备
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()  # Set to inference mode
    print("Model loaded successfully!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """生成文本"""
    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # 生成参数
    generation_config = {
        "max_new_tokens": max_new_tokens,       # 模型生成的token数量
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,  # 减少重复
    }
    
    # 记录生成时间
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(inputs, **generation_config)
    
    generation_time = time.time() - start_time
    
    # 解码输出（只返回新生成的部分）
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    return generated_text, generation_time

def test_model_capabilities(model, tokenizer):
    """测试模型的各种能力"""
    
    test_prompts = [
        # 1. Basic text completion
        "The history of artificial intelligence can be divided into",
        "Deep learning is a branch of machine learning that",
        
        # 2. Common sense reasoning
        "The sun rises in the east and sets in the",
        "Water boils at 100 degrees",
        
        # 3. Simple math
        "2 + 3 = ",
        "10 multiplied by 5 equals",
        
        # 4. Logical reasoning
        "If today is Monday, then tomorrow is",
        "All birds can fly. Penguins are birds. Therefore, penguins",
        
        # 5. Creative writing
        "Once upon a time, there was a mountain, and on the mountain there was a",
        "Spring has come, flowers are blooming, and",
        
        # 6. Question answering
        "What is machine learning?",
        "Please explain how neural networks work.",
        
        # 7. Conversation
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    
    print("=" * 60)
    print("Starting model inference capability testing")
    print("=" * 60)
    
    total_time = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【Test {i}】")
        print(f"Input: {prompt}")
        print("-" * 40)
        
        # 生成文本
        generated, gen_time = generate_text(
            model, tokenizer, prompt, 
            max_new_tokens=80, 
            temperature=0.7,
            top_p=0.9
        )
        
        total_time += gen_time
        
        print(f"Output: {generated}")
        print(f"Generation time: {gen_time:.2f}s")
        print("-" * 40)
    
    print(f"\nTotal test time: {total_time:.2f}s")
    print(f"Average generation time: {total_time/len(test_prompts):.2f}s")

def interactive_chat(model, tokenizer):
    """交互式对话测试"""
    print("\n" + "=" * 60)
    print("Enter interactive mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
            
        print("Model generating...")
        generated, gen_time = generate_text(
            model, tokenizer, user_input,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"Model: {generated}")
        print(f"(Generation time: {gen_time:.2f}s)")

def main():
    """主函数"""
    # 模型路径 - 如果训练完成，使用output根目录；否则使用最新checkpoint
    model_path = "../output" 
    
    try:
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Display model info
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Device: {model.device}")
        
        # Automated testing
        # test_model_capabilities(model, tokenizer)
        
        # Interactive testing
        interactive_chat(model, tokenizer)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the model path is correct")

if __name__ == "__main__":
    main()