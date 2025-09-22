#!/usr/bin/env python3
# 奖励模型验证脚本
# 2025/9/10
# wenzhe

import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
)

# 设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'


def load_reward_model(reward_model_path="./reward_model_output", tokenizer_path="../tokenizer/byte_level_bpe_tokenizer_v1.json"):
    """加载完整的奖励模型"""
    print("🔄 加载分词器...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>", 
        pad_token="<pad>", 
        eos_token="</s>", 
        unk_token="<unk>",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🔄 加载完整奖励模型...")
    # 直接加载完整的奖励模型（已经merge的）
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    
    print("✅ 奖励模型加载完成！")
    return model, tokenizer


def get_reward_score(model, tokenizer, instruction, response):
    """计算奖励分数"""
    text = f"Instruction: {instruction}\nResponse: {response}"
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        reward_score = outputs.logits.squeeze().cpu().item()
    
    return reward_score


def test_reward_model():
    """测试奖励模型"""
    try:
        # 加载模型
        model, tokenizer = load_reward_model()
        
        # 测试用例
        test_cases = [
            {
                "instruction": "解释什么是Python编程语言",
                "good_response": "Python是一种高级、解释型编程语言，以其简洁易读的语法著称。它支持面向对象、函数式和过程式编程范式，广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。Python拥有丰富的标准库和第三方包生态系统。",
                "bad_response": "Python就是蛇。"
            },
            {
                "instruction": "如何学习机器学习？",
                "good_response": "学习机器学习建议按以下步骤：1.掌握数学基础(线性代数、概率统计、微积分)；2.学习Python编程和相关库(numpy、pandas、scikit-learn)；3.理解基本算法原理；4.通过实际项目练习；5.参加在线课程和阅读经典教材；6.关注最新研究进展。",
                "bad_response": "看书就行了。"
            },
            {
                "instruction": "什么是深度学习？",
                "good_response": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。通过大量数据训练，深度神经网络能够自动学习数据的层次化特征表示。它在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。",
                "bad_response": "深度学习就是很深的学习。"
            }
        ]
        
        print("\n🎯 奖励模型验证结果：")
        print("=" * 50)
        
        correct_count = 0
        total_count = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n测试 {i}: {case['instruction']}")
            
            good_score = get_reward_score(model, tokenizer, case['instruction'], case['good_response'])
            bad_score = get_reward_score(model, tokenizer, case['instruction'], case['bad_response'])
            
            print(f"  详细回答分数: {good_score:.4f}")
            print(f"  简单回答分数: {bad_score:.4f}")
            print(f"  分数差异: {good_score - bad_score:.4f}")
            
            if good_score > bad_score:
                print("  ✅ 正确识别")
                correct_count += 1
            else:
                print("  ❌ 识别错误")
        
        accuracy = correct_count / total_count
        print(f"\n📊 总体结果:")
        print(f"  正确识别: {correct_count}/{total_count}")
        print(f"  准确率: {accuracy:.2%}")
        
        if accuracy >= 0.8:
            print("  🎉 奖励模型表现优秀！")
        elif accuracy >= 0.6:
            print("  👍 奖励模型表现良好")
        else:
            print("  ⚠️  奖励模型可能需要进一步优化")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        print("\n请检查:")
        print("1. 奖励模型训练是否完成")
        print("2. 完整奖励模型路径: ./reward_model_output/")
        print("3. 分词器路径: ../tokenizer/")


if __name__ == "__main__":
    test_reward_model()
