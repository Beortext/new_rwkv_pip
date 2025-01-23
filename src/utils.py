import torch, copy, re, os

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class Tokenize():
    def __init__(self, file_name=None):
        """
        Function: Initializes the tokenizer and loads the vocabulary file.
        Parameters:
            - file_name (str, optional): Path to the vocabulary file.
            Defaults to rwkv_vocab_v20230424.txt in the same directory.
        """
        self.idx2token = {}

        if file_name == None:
            file_name = os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt'
        
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encode(self, src):
        """
        Function: Encodes a UTF-8 string into a token sequence.
        Parameters:
            - src (str): Input text.
        Returns:
            - list[int]: Token sequence.
        Algorithm:
            - Performs greedy longest-match tokenization using the TRIE tree.
        """
        idx:int = 0
        tokens = []
        src = src.encode("utf-8")

        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)

        return tokens

    def decode(self, tokens):
        """
        Function: Decodes a token sequence into text.
        Parameters:
            - tokens (list[int]): Token sequence.
        Returns:
            - str: Decoded text, with invalid bytes replaced by Unicode placeholder `\ufffd`.
        """
        try:
            return b''.join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8

class Pipeline():
    def __init__(self, model, tokenizer_file_name=None):
        """
        Function: Initializes the inference pipeline.
            - Parameters:
            - model: Language model instance.
            - tokenizer_file_name: Path to the tokenizer vocabulary file.
        """
        self.model_tokens = []
        self.init_state = None
        self.model = model
        self.tokenizer = Tokenize(tokenizer_file_name)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        """
        Function: Samples logits using configurable decoding strategies.
        Parameters:
            - temperature (float): Sampling temperature (>1 increases randomness, <1 reduces).
            - top_p (float): Nucleus sampling threshold (cumulative probability cutoff).
            - top_k (int): Number of top candidates to retain.
        Returns:
            - int: Sampled token ID.
        """
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
        """
        Function: Loads model states from disk.
        Parameters:
            - state_file (str): Path to state file (automatically appends .pth if missing).
        """
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
        """
        Function: Deletes the currently stored model states.
        """
        self.init_state = None
    
    def prefill(self, ctx, states:list=None):
        """
        Function: Prefills context into the model.
        Parameters:
            - ctx (str): Context text.
            - states: Initial model states.
        Returns:
            - Updated model states.
        """
        if self.init_state == None:
            states =  copy.deepcopy(self.init_state)

        tokens = self.tokenizer.encode(ctx)
        _, states =self.model.forward(tokens, states)
        
        return states

    def generate(self, content:str, gen_len=4096, states:list=None, temperature=1.0, top_p=0.3, top_k=0, presence=0.5, frequency=0.5, penalty_decay = 0.996):
        """
        Function: Executes text generation.
        Key Parameters:
            - presence/frequency (float): Repetition penalty coefficients.
            - penalty_decay (float): Penalty decay factor.
        Features:
            - Real-time printing of generated content.
            - Early termination on double newline detection.
            - Automatic handling of UTF-8 encoding boundaries.
        """
        if self.init_state != None:
            top_p = 0.2
            presence = 0.3
            frequency = 0.3
            states = copy.deepcopy(self.init_state)
        
        if len(content) > 0:
            occurrence = {}
            out_tokens = []
            out_last = 0

            tokens = self.tokenizer.encode(content)
            tokens = [int(x) for x in tokens]
            gen_len -= len(tokens)
            out, states =self.model.forward(tokens, states)

            for i in range(gen_len):
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

def chat_prompt(user_name = "User", assistant_name = "Assistant", out_format: str = "User: {}\n\nAssistant:"):
    """
    Function: Generates formatted dialogue prompts.
    Parameters:
        - user_name (str): User identifier.
        - assistant_name (str): Assistant identifier.
        - out_format (str): Output format template (default: "User: {}\n\nAssistant:").
    """
    msg = input(f"{user_name}: ")

    if msg.strip():
        msg = re.sub(r"\n+", "\n", msg)
        print(f"\n{assistant_name}:", end="")

        return out_format.format(msg)
    else:
        return ""