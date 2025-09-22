
# 🚀 从零构建1亿参数大语言模型：完整的预训练与微调流程

本项目实现了一个完整的大语言模型训练流程，从自定义分词器训练开始，基于GPT-2架构进行1亿参数规模的预训练，并采用先进的微调技术（QLoRA + RLHF）来提升模型性能。

## 🎯 项目概述

### 核心特性

- **从零开始**: 完整的模型训练流程，包括分词器训练、预训练、有监督微调和强化学习微调
- **1亿参数规模**: 适中的模型规模，平衡了性能与计算资源需求
- **先进技术栈**:
  - DeepSpeed分布式训练加速
  - QLoRA量化微调技术
  - RLHF人类偏好对齐
  - 自定义BPE分词器
- **完整工程化**: 支持断点续训、分布式训练、显存优化等生产级特性

### 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   分词器训练     │───▶│    模型预训练    │───▶│   有监督微调     │───▶│   强化学习微调   │
│  (Tokenizer)   │    │   (PreTrain)   │    │     (SFT)      │    │    (RLHF)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 项目结构

```
llm_train/
├── tokenizer/                    # 分词器相关
│   └── byte_level_bpe_tokenizer_v1.json
├── PreTrain/                     # 预训练阶段
│   ├── train_byte_level_bpe_tokenizer.py  # 分词器训练
│   ├── preprocess_dataset_for_pretrain_map.py  # 数据预处理
│   ├── model.py                  # GPT-2模型定义
│   ├── ds_config.json           # DeepSpeed配置
│   └── inference.py             # 预训练模型推理
├── SFT/                         # 有监督微调
│   ├── qlora_fine_tune_trl.py   # QLoRA微调脚本
│   ├── inference.py             # 微调模型推理
│   └── sft_output/              # SFT输出模型
├── RLHF/                        # 强化学习微调
│   ├── reward_model.py          # 奖励模型训练
│   ├── grpo_trl.py              # GRPO算法实现
│   └── grpo_training.py         # GRPO训练脚本
├── output/                      # 预训练模型输出
└── pretrain_data_chunks/        # 预处理数据块
```

## 🛠️ 环境配置

### 系统要求

- Python 3.10+
- CUDA 11.8+
- 16GB+ GPU显存（推荐多卡）
- 100GB+ 磁盘空间

### 依赖安装

```bash
# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install deepspeed peft trl bitsandbytes
pip install tokenizers psutil

# 可选依赖
pip install wandb tensorboard  # 训练监控
pip install gradio streamlit   # 模型部署
```

## 🚀 训练流程

### 第一阶段：分词器训练

```bash
cd PreTrain/
python train_byte_level_bpe_tokenizer.py
```

- 基于大规模语料训练BPE分词器
- 词汇表大小：50k
- 支持中英文及代码文本

### 第二阶段：数据预处理

```bash
python preprocess_dataset_for_pretrain_map.py
```

- 数据源：C4、Wikipedia、Tiny Codes、Gutenberg、StackExchange
- 自动分块、分词，生成parquet文件
- 支持多进程加速处理

### 第三阶段：模型预训练

```bash
# 单卡训练
python -m torch.distributed.launch --nproc_per_node=1 \
    --use_env model.py

# 多卡训练（推荐）
deepspeed --num_gpus=4 model.py --deepspeed ds_config.json
```

- 模型架构：GPT-2 (12层, 768维, 12头)
- 参数量：~1亿
- 训练数据：数十GB文本数据
- 优化器：AdamW + 学习率调度

### 第四阶段：有监督微调 (SFT)

```bash
cd ../SFT/
accelerate launch --num_processes=4 qlora_fine_tune_trl.py
```

- 技术栈：QLoRA + 4bit量化
- 数据集：高质量指令-回答对
- 显存优化：支持单卡训练大模型

### 第五阶段：强化学习微调 (RLHF)

```bash
cd ../RLHF/

# 1. 训练奖励模型
python reward_model.py

# 2. GRPO强化学习训练
accelerate launch --num_processes=4 grpo_trl.py
```

- 奖励模型：基于人类偏好数据训练
- GRPO算法：改进的PPO算法，更稳定
- 人类偏好对齐：提升模型输出质量

## 📊 模型性能

### 预训练模型

- **参数量**: ~1亿
- **词汇表**: 50k tokens
- **上下文长度**: 1024 tokens
- **训练数据**: 数十GB多语言文本

### 微调效果

- **SFT后**: 指令跟随能力显著提升
- **RLHF后**: 输出质量和安全性进一步改善
- **推理速度**: 单卡T4可实时推理

## 🔧 使用示例

### 模型推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("./output")

# 加载微调适配器
model = PeftModel.from_pretrained(base_model, "./SFT/sft_output")

# 推理示例
prompt = "请介绍一下人工智能的发展历程"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 模型验证

```bash
# 预训练模型推理
cd PreTrain/
python inference.py

# SFT模型推理
cd ../SFT/
python inference.py

# RLHF模型推理
cd ../RLHF/
python grpo_trl.py --mode inference
```

## 📈 训练监控

### 关键指标

- **Loss曲线**: 训练/验证损失变化
- **学习率**: 自适应学习率调度
- **显存使用**: 实时显存监控
- **吞吐量**: tokens/秒处理速度

### 日志文件

- `train.log`: 预训练日志
- `SFT/nohup.out`: 微调训练日志  
- `RLHF/train.log`: 强化学习日志

## 🎛️ 配置说明

### DeepSpeed配置 (`ds_config.json`)

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 5e-5}
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true
  }
}
```

### QLoRA配置

- **量化**: 4bit NF4
- **LoRA秩**: r=64
- **学习率**: 1e-4
- **批次大小**: 4

### RLHF配置  

- **算法**: GRPO (Group Relative Policy Optimization)
- **β参数**: 0.05
- **奖励缩放**: 关闭
- **训练轮数**: 3-5 epochs

## 🚨 常见问题

### Q: 显存不足怎么办？

A:

- 减小batch_size
- 启用gradient_checkpointing
- 使用更高的量化精度
- 尝试DeepSpeed ZeRO-3

### Q: 训练中断如何恢复？

A: 所有训练脚本都支持断点续训，会自动从最新checkpoint恢复

### Q: 如何评估模型效果？

A: 使用对应的inference脚本进行模型验证和对比测试

### Q: RLHF模型输出乱码怎么办？

A: 检查分词器一致性，确保GRPO训练时`add_special_tokens=True`

## 🔍 技术细节

### 预训练优化

- **混合精度**: BF16训练加速
- **激活检查点**: 减少显存占用
- **梯度累积**: 模拟大批次训练
- **参数卸载**: CPU offload节省显存

### QLoRA优化

- **4bit量化**: NF4量化减少显存
- **LoRA适配器**: 低秩适配高效微调
- **梯度检查点**: 进一步节省显存
- **数据并行**: 多卡加速训练

### RLHF优化

- **奖励模型**: 人类偏好建模
- **GRPO算法**: 稳定的策略优化
- **KL正则化**: 防止偏离基础模型
- **经验回放**: 提高样本利用率

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发规范

- 代码风格：遵循PEP8
- 提交格式：使用conventional commits
- 测试：确保推理脚本正常运行

### 后续计划

- [ ] 添加模型评估指标
- [ ] 支持更多数据集
- [ ] 优化训练效率
- [ ] 添加Web界面

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- Hugging Face Transformers
- Microsoft DeepSpeed  
- TRL (Transformer Reinforcement Learning)
- OpenAI GPT-2

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
