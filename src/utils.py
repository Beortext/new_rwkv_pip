import torch, copy

from .rwkv_tokenizer import TRIE_TOKENIZER

class Pipline():
    def __init__(self, model, tokenizer_file_name=None):
        self.model_tokens = []
        self.init_state = None
        self.model = model
        self.tokenizer = TRIE_TOKENIZER(tokenizer_file_name)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
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
    
    def load_state(self, state_file:str):
        state_raw = torch.load(state_file if '.pth' in state_file else state_file + '.pth', weights_only=True)
        state_init = [None for _ in range(self.model.n_layer * 3)]
        for i in range(self.model.n_layer):
            dd = self.model.strategy[i]
            dev = dd.device
            atype = dd.atype    
            state_init[i*3+0] = torch.zeros(self.model.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
            state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
            state_init[i*3+2] = torch.zeros(self.model.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        self.init_state = copy.deepcopy(state_init)

    def del_state(self):
        self.init_state = None
    
    def prefill(self, ctx, states:list=None):
        if self.init_state != None:
            states =  copy.deepcopy(self.init_state)

        tokens = self.tokenizer.encode(ctx)
        _, states =self.model.forward(tokens, states)
        
        return states

    def generate(self, ctx:str, ctx_len=4096, states:list=None, temperature=1.0, top_p=0.85, top_k=0, presence=0.5, frequency=0.5, penalty_decay = 0.996):
        if self.init_state != None:
            top_p = 0.2
            presence = 0.3
            frequency = 0.3
            states = copy.deepcopy(self.init_state)
        
        if len(ctx) > 0:
            occurrence = {}
            out_tokens = []
            out_last = 0

            tokens = self.tokenizer.encode(ctx)
            tokens = [int(x) for x in tokens]
            ctx_len -= len(tokens)
            out, states =self.model.forward(tokens, states)

            for i in range(ctx_len):
                for n in occurrence:
                    out[n] -= presence + occurrence[n] * frequency # repetition penalty
                out[0] -= 1e10  # disable END_OF_TEXT

                token = self.sample_logits(out, temperature, top_p, top_k)
                out, states = self.model.forward(token, states)
                out_tokens += [token]

                for n in occurrence:
                    occurrence[n] *= penalty_decay
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

                tmp = self.tokenizer.decode(out_tokens[out_last:])
                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                    print(tmp, end="", flush=True)
                    out_last = i + 1

                if "\n\n" in tmp:
                    print(tmp, end="", flush=True)
                    break

            return states
        else:
            print("!!! Error: please say something !!!")