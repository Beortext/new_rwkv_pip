########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print("RWKV Chat Simple Demo")

import os, re
from prompt_toolkit import prompt


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"  # !!! '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries !!!

os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1" # only support Tesla P40.

from src.model import RWKV
from src.utils import Pipline

########################################################################################################

name = 'RWKV-5-World-1B5-v2-20231025-ctx4096'
# RWKV-4-World-0.1B-v1-20230520-ctx4096 RWKV-5-World-1B5-v2-20231025-ctx4096
# RWKV-x060-World-1B6-v2.1-20240328-ctx4096 RWKV-x070-World-0.1B-v2.8-20241210-ctx4096

strategy = "cuda fp16"  # use CUDA, fp16
MODEL_NAME = f"/home/beortust/nfs_share/RwkvModelLib/RWKV-x060-ChnNovel-7B-20240803-ctx4096.pth"

########################################################################################################
STATE_NAME = None # use vanilla zero initial state?

# use custom state? much better chat results (download from https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)
# note: this is English Single-round QA state (will forget what you previously say)
# STATE_NAME = "E://RWKV-Runner//models//rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048"
########################################################################################################

print(f"Loading model - {MODEL_NAME}")
model = RWKV(model_path=MODEL_NAME, strategy=strategy)
pipline = Pipline(model)
# pipline.load_state("/home/beortust/nfs_share/RwkvModelLib/state-tuna/rwkv-x060-OnlyForChnNovel_小说扩写-7B-20240806-ctx4096.pth")

states = pipline.prefill("User: hi\n\nAssistant: Hello, I'am your AI Assistant. Are your question?")

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    print("\nAssistant:", end='')
    states = pipline.generate(f"User: {msg}\n\nAssistant:", 500, states)
