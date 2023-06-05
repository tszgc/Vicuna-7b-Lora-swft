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
parser.add_argument("--model_path", type=str, default="/Users/nina/zoo/document4work/SWFT/LLM/vicuna-7b")
parser.add_argument("--lora_path", type=str, default="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco")
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()
print(args)

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')

device="cuda"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    #load_in_8bit=LOAD_8BIT,
    torch_dtype=torch.float16,
    device_map="auto", #device_map={"": 0},
)
model = StreamPeftGenerationMixin.from_pretrained(
    model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map="auto", #device_map={"": 0}
)




if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    **kwargs,
):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = tokenizer.decode(output).strip()
        # output = tokenizer.decode(output).split("### Response:")[1].strip()
        print(output)
        yield output

def test():
    print('开始测试')
    input = "###-TASK-A-A-A, no matter feasibility, answer only one word, 'positive' or 'negative', by this sentence:Blockware\u2019s team expects Bitcoin\u2019s adoption rate to be faster than previous technologies, but believes it's still in early-stage growth.\\xa0";
    output = evaluate(input)
    print(output)

if __name__ == "__main__":
    test()
