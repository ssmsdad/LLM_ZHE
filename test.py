

import os
os.environ['HF_DATASETS_CACHE'] = '/data/wenzhe/huggingface/datasets'
os.environ['HF_HOME'] = '/data/wenzhe/huggingface'
from datasets import load_dataset

from datasets import load_dataset

ds = load_dataset("vicgalle/alpaca-gpt4")

print(ds['train'][0])



