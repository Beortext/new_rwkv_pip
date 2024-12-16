########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print("RWKV Chat Simple Demo")

import os, copy, types, re
from prompt_toolkit import prompt
import torch


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"  # !!! '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries !!!


os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1"

from src.model import RWKV
from src.rwkv_tokenizer import TRIE_TOKENIZER

########################################################################################################

name = 'RWKV-x070-World-0.1B-v2.8-20241210-ctx4096'
# RWKV-4-World-0.1B-v1-20230520-ctx4096 RWKV-5-World-1B5-v2-20231025-ctx4096
# RWKV-x060-World-1B6-v2.1-20240328-ctx4096 RWKV-x070-World-0.1B-v2.8-20241210-ctx4096

args = types.SimpleNamespace()
args.strategy = "cuda fp16"  # use CUDA, fp16
args.MODEL_NAME = f"//home//beortust//nfs_share//RwkvModelLib//test-models//{name}"

########################################################################################################
STATE_NAME = None # use vanilla zero initial state?

# use custom state? much better chat results (download from https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)
# note: this is English Single-round QA state (will forget what you previously say)
# STATE_NAME = "E://RWKV-Runner//models//rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048"
########################################################################################################

GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.5
GEN_alpha_frequency = 0.5
GEN_penalty_decay = 0.996

if STATE_NAME != None:
    GEN_TOP_P = 0.2
    GEN_alpha_presence = 0.3
    GEN_alpha_frequency = 0.3

CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model_path=args.MODEL_NAME, strategy=args.strategy)
tokenizer = TRIE_TOKENIZER()

model_tokens = []
model_state = None

if STATE_NAME != None: # load custom state
    args = model.args
    state_raw = torch.load(STATE_NAME if '.pth' in STATE_NAME else STATE_NAME + '.pth', weights_only=True)
    state_init = [None for i in range(args.n_layer * 3)]
    for i in range(args.n_layer):
        dd = model.strategy[i]
        dev = dd.device
        atype = dd.atype    
        state_init[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
        state_init[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    model_state = copy.deepcopy(state_init)

def run_rnn(ctx):
    global model_tokens, model_state

    ctx = ctx.replace("\r\n", "\n")

    tokens = tokenizer.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    out, model_state = model.forward(tokens, model_state)

    return out

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_ids = torch.argsort(probs)
    sorted_probs = probs[sorted_ids]
    sorted_probs = torch.flip(sorted_probs, dims=(0,))
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = torch.nonzero(cumulative_probs >= top_p, as_tuple=True)[0]
    if len(cutoff_index) > 0:
        cutoff = float(sorted_probs[cutoff_index[0]])
    else:
        cutoff = float(sorted_probs[-1])
    probs[probs < cutoff] = 0
    if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / torch.sum(probs)
    logits = torch.multinomial(probs, num_samples=1)[0]
    
    return int(logits)

if STATE_NAME == None: # use initial prompt if we are not loading a state
    init_ctx = "User: hi" + "\n\n"
    init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"
    run_rnn(init_ctx)
    print(init_ctx, end="")

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    if len(msg) > 0:
        occurrence = {}
        out_tokens = []
        out_last = 0

        out = run_rnn("User: " + msg + "\n\nAssistant:")
        print("\nAssistant:", end="")

        for i in range(4096):
            for n in occurrence:
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
            out[0] -= 1e10  # disable END_OF_TEXT

            token = sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)

            out, model_state = model.forward([token], model_state)
            model_tokens += [token]

            out_tokens += [token]

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = tokenizer.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break
    else:
        print("!!! Error: please say something !!!")
