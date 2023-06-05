import sys
import torch
import transformers
import argparse
import warnings
import os
from utils import StreamPeftGenerationMixin, StreamLlamaForCausalLM

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/model/13B_hf")
parser.add_argument("--lora_path", type=str, default="checkpoint-3000")
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()
print(args)

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

if __name__ == "__main__":
    print('main')
