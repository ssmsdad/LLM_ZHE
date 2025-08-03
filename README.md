
# GPT-2大模型分布式预训练（DeepSpeed加速）

本项目实现了基于DeepSpeed的GPT-2大规模预训练，支持自定义分词器、超大数据集分块（千万级别）、显存优化和断点续训。

## 环境准备

我这里使用Python 3.10.18，安装如下依赖：

```bash
pip install deepspeed transformers datasets torch psutil
```

## 数据预处理

1. 数据集自动加载（C4、Wikipedia、Tiny Codes、Gutenberg、StackExchange），并拼接、打乱。
2. 分词、分块，保存为parquet文件，支持多进程/单进程高效处理。

运行预处理脚本：

```bash
python preprocess_dataset_for_pretrain.py
```

## 训练分词器

我这里训练了一个byte-level bpe分词器，详细代码见 `train_byte_level_bpe_tokenizer.py`

运行训练分词器脚本：

```bash
python train_byte_level_bpe_tokenizer.py
```

## 训练模型

训练主流程见 `model.py`，支持DeepSpeed分布式加速、BF16混合精度、ZeRO Stage 2优化、参数/优化器状态CPU offload。

启动训练命令（4卡示例）：

```bash
deepspeed --num_gpus=4 model.py
```

模型参数（约1B）：
- vocab_size=32000
- n_positions=2048
- n_embd=2048
- n_layer=24
- n_head=16

训练参数（可在`ds_config.json`中调整）：
- per_device_train_batch_size=8
- gradient_accumulation_steps=16
- bf16=True
- zero_optimization.stage=2
- zero_optimization.offload_param.device=cpu
- activation_checkpointing.partition_activations=True

## ds_config.json 说明

- `zero_optimization.stage`: ZeRO优化阶段，2为主流大模型推荐。
- `offload_param.device`: 参数offload到CPU，显存不足时建议开启。
- `activation_checkpointing`: 激活值分区与连续内存优化，进一步降低显存占用。
- `bf16`: 推荐使用BF16混合精度，3090/4090等新卡支持。

## 断点续训与模型保存

训练过程自动保存模型和分词器到`./output`目录，支持断点续训。

## 其他说明

- 数据分块、分词、保存均已高度优化，支持千万级数据(2TB)。
- 训练脚本兼容HuggingFace Trainer和DeepSpeed。

## 训练硬件

- 4张3090/4090显卡，单卡24GB显存。
- 内存256GB。
