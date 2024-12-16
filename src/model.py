########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, gc, os, re, os
import torch
from torch.nn import functional as F

########################################################################################################

if os.environ.get('RWKV_JIT_ON') == '1':
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    MyFunction = lambda x: x
    MyStatic = lambda x: x

if os.environ.get('RWKV_DML_ON') == '1':
    import torch_directml
    print("PyTorch with DirectML Enabled")

########################################################################################################

class RWKV(MyModule):
    def __init__(self, model_path, strategy, is_verbose = True, convert_and_save_and_exit = None):
        super().__init__()
        if is_verbose:
            verbose_out = lambda *args, **kwargs: print(*args, **kwargs)
        else:
            verbose_out = lambda *args, **kwargs: None

        STRATEGY_REGEX = r"^(?:(?:^|->) *(?:cuda(?::[\d]+)?|cpu|mps|dml) (?:fp(?:16|32)|bf16)(?:i8|i4|i3)?(?: \*[\d]+\+?)? *)+$"
        if not re.match(STRATEGY_REGEX, strategy):
            raise ValueError("Invalid strategy. Please read https://pypi.org/project/rwkv/")

        strategy = ('->'.join([x.strip() for x in strategy.split('->')])).replace('->', ' -> ')
        self.strategy_string = strategy

        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
        try:
            self.rescale_layer = int(os.environ["RWKV_RESCALE_LAYER"]) # !!! NOTE: SEEMS YOU SHOULD SET IT TO 999 (disable) FOR RWKV-MUSIC MODELS !!!
        except:
            self.rescale_layer = 6 if 'fp16' in strategy else 0

        verbose_out(f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.rescale_layer}\n')

        model_path = model_path.strip()
        if not model_path.endswith('.pth'):
            model_path += '.pth'
        verbose_out(f'Loading {model_path} ...')

        with torch.no_grad():
            self.model_parm = torch.load(model_path, map_location='cpu', weights_only=True) # load model to CPU first
            gc.collect()

            ALREADY_CONVERTED = False
            if '_strategy' in self.model_parm:
                ALREADY_CONVERTED = True
                assert convert_and_save_and_exit == None # you should only convert a raw model
                verbose_out(f"Converted model: strategy {self.model_parm['_strategy']}\n")
                assert self.model_parm['_strategy'] == self.strategy_string # if you are using a new strategy, re-convert the model
                # assert float(self.model_parm['_version']) >= 0.7 # sometimes you should re-convert using latest convert_model.py
                assert self.model_parm['_rescale_layer'] == self.rescale_layer # must use same RESCALE_LAYER to avoid mistakes
                del self.model_parm['_strategy']
                # del self.model_parm['_version']
                del self.model_parm['_rescale_layer']
            
            self.n_embd = self.model_parm['emb.weight'].shape[1]
            self.n_layer = 0

            keys = list(self.model_parm.keys())
            self.version = 4
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                self.n_layer = max(self.n_layer, layer_id+1)
                if 'ln_x' in x:
                    self.version = max(5, self.version)
                if 'gate.weight' in x:
                    self.version = max(5.1, self.version)
                if int(self.version) == 5 and 'att.time_decay' in x:
                    self.n_head = self.model_parm[x].shape[0]
                    if len(self.model_parm[x].shape) > 1:
                        if self.model_parm[x].shape[1] > 1:
                            self.version = max(5.2, self.version)
                if 'time_maa' in x:
                    self.version = max(6, self.version)
                if int(self.version) == 6 and 'time_faaaa' in x:
                    self.n_head = self.model_parm[x].shape[0]

                if 'r_k' in x:
                    self.version = max(7, self.version)
                    self.n_head = self.model_parm[x].shape[0]
            verbose_out(f'Model detected: v{self.version:.1f}')

            ####################### Compute strategy

            strategy_rules_list = [x.strip().split(' ') for x in strategy.split('->')]
            allocate_plan = [0 for _ in range(len(strategy_rules_list))]
            is_tmp_layer_i = -1
            is_tmp_layer_count = 0
            to_allocate = self.n_layer + 1
            allocated = 0
            free_slots = 0
            for i in range(len(strategy_rules_list)):
                rules = strategy_rules_list[i]
                dtype = rules[1]
                if dtype.startswith('fp32'): rules[1] = [torch.float]
                elif dtype.startswith('fp16'): rules[1] = [torch.float16]
                elif dtype.startswith('bf16'): rules[1] = [torch.bfloat16]
                else: exit(f"{dtype} type is not supported.")

                if dtype.endswith('i8'): rules[1] += [torch.uint8]
                else: rules[1] += [rules[1][0]]

                if len(rules) > 2:
                    ss = rules[2]
                    assert ss.startswith('*')
                    if ss.endswith('+'):
                        allocate_plan[i] = int(ss[1:-1])
                        is_tmp_layer_i = i
                    else:
                        allocate_plan[i] = int(ss[1:])
                    allocated += allocate_plan[i]
                    if allocated >= to_allocate:
                        allocate_plan[i] += to_allocate - allocated
                        break
                else:
                    free_slots += 1
                    
            if is_tmp_layer_i < 0:
                if free_slots > 0 and to_allocate > allocated:
                    for i in range(len(strategy_rules_list)):
                        if allocate_plan[i] == 0:
                            allocate_plan[i] = (to_allocate - allocated) // free_slots
                            allocated += allocate_plan[i]
                            free_slots -= 1
                if to_allocate > allocated:
                    allocate_plan[len(strategy_rules_list)-1] += to_allocate - allocated
            else:
                if to_allocate > allocated:
                    is_tmp_layer_count = to_allocate - allocated
                    allocate_plan[is_tmp_layer_i] += is_tmp_layer_count

            verbose_out(f'Strategy: (total {self.n_layer}+1={self.n_layer+1} layers)')
            for i in range(len(strategy_rules_list)):
                rules = strategy_rules_list[i]
                if i != is_tmp_layer_i:
                    verbose_out(f'* {rules[0]} {str(rules[1]).replace("torch.","")}, Store {allocate_plan[i]} layers')
                else:
                    verbose_out(f'* {rules[0]} {str(rules[1]).replace("torch.","")}, Store {allocate_plan[i]-is_tmp_layer_count} layers, Tmp {is_tmp_layer_count} layers')
                allocate_plan[i] += (0 if i == 0 else allocate_plan[i-1])

            self.strategy = [None] * (self.n_layer + 1)
            strategy = self.strategy
            for n in range(self.n_layer + 1):
                for i in range(len(strategy_rules_list)):
                    if n < allocate_plan[i]:
                        strategy[n] = types.SimpleNamespace()
                        strategy[n].device = strategy_rules_list[i][0]
                        strategy[n].atype = strategy_rules_list[i][1][0]
                        strategy[n].wtype = strategy_rules_list[i][1][1]
                        strategy[n].is_tmp_layer = False
                        if strategy[n].device == 'dml':
                            strategy[n].device = torch_directml.device()
                        if i == is_tmp_layer_i and n >= (allocate_plan[i] - is_tmp_layer_count):
                            strategy[n].is_tmp_layer = True
                        break
                if n-1 > 0 and strategy[n].device != strategy[n-1].device: verbose_out()
                verbose_out(f"{n}-{strategy[n].device}-{str(strategy[n].atype).replace('torch.','')}-{str(strategy[n].wtype).replace('torch.','')}{'-Tmp' if strategy[n].is_tmp_layer else ''}",end=' ')
            verbose_out()

            ####################### Load weights to self.w

            if not ALREADY_CONVERTED:
                try: # precompute embedding
                    self.model_parm['emb.weight'] = F.layer_norm(self.model_parm['emb.weight'], (self.n_embd,), weight=self.model_parm['blocks.0.ln0.weight'], bias=self.model_parm['blocks.0.ln0.bias'])
                except:
                    self.model_parm['emb.weight'] = F.layer_norm(self.model_parm['emb.weight'].float(), (self.n_embd,), weight=self.model_parm['blocks.0.ln0.weight'].float(), bias=self.model_parm['blocks.0.ln0.bias'].float())
                    
                if self.version >= 7.0:
                    self.model_parm['blocks.0.att.v0'] = self.model_parm['blocks.0.att.a0'] # actually ignored
                    self.model_parm['blocks.0.att.v1'] = self.model_parm['blocks.0.att.a1'] # actually ignored
                    self.model_parm['blocks.0.att.v2'] = self.model_parm['blocks.0.att.a2'] # actually ignored

                del self.model_parm['blocks.0.ln0.weight']
                del self.model_parm['blocks.0.ln0.bias']

            print_need_newline = True

            REAL_TIME_FIRST = False
            self.time_state = False
            for x in list(self.model_parm.keys()):
                if '.time_faaaa' in x: REAL_TIME_FIRST = True
                if '.time_state' in x: self.time_state = True
            if REAL_TIME_FIRST:
                self.model_parm = {k.replace('.time_faaaa','.time_first') if '.time_faaaa' in k else k: v for k, v in self.model_parm.items()}
                self.model_parm = self.model_parm
            
            keys = list(self.model_parm.keys())
            for x in keys:
                self.model_parm[x].requires_grad = False
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                if ('ln_out.' in x) or ('head.' in x):
                    layer_id = self.n_layer
                dd = strategy[layer_id]
                DEVICE = dd.device
                ATYPE = dd.atype
                WTYPE = dd.wtype

                if not ALREADY_CONVERTED:
                    if self.rescale_layer > 0:
                        if 'att.output.weight' in x:
                            self.model_parm[x] = self.model_parm[x] / (2 ** int(layer_id // self.rescale_layer))
                        if 'ffn.value.weight' in x:
                            self.model_parm[x] = self.model_parm[x] / (2 ** int(layer_id // self.rescale_layer))

                    if '.time_' in x:
                        self.model_parm[x] = self.model_parm[x].squeeze()
                    if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'gate.weight' in x or 'output.weight' in x or 'head.weight' in x:
                        self.model_parm[x] = self.model_parm[x].t()

                    if '.time_decay' in x and '_w' not in x: # need fp32 for this
                        if self.version == 4:
                            self.model_parm[x] = -torch.exp(self.model_parm[x].float())
                        elif int(self.version) == 5:
                            self.model_parm[x] = torch.exp(-torch.exp(self.model_parm[x].float())).reshape(-1,1,1)
                            if self.version == 5.2:
                                self.model_parm[x] = self.model_parm[x].reshape(self.n_head, -1, 1)
                        elif self.version == 6.0:
                            self.model_parm[x] = self.model_parm[x].float().reshape(self.n_head, -1, 1)
                    elif '.time_first' in x: # need fp32 for this
                        if self.version == 4:
                            self.model_parm[x] = self.model_parm[x].float()
                        elif int(self.version) in [5, 6]:
                            if REAL_TIME_FIRST:
                                self.model_parm[x] = self.model_parm[x].float().reshape(-1,1,1)
                            else:
                                self.model_parm[x] = torch.exp(self.model_parm[x].float()).reshape(-1,1,1)
                            if self.version in [5.2, 6.0]:
                                self.model_parm[x] = self.model_parm[x].reshape(self.n_head, -1, 1)
                    elif 'att.r_k' in x: 
                        self.model_parm[x] = self.model_parm[x].flatten()
                    elif '.ln_x' in x: # need fp32 for group_norm
                        self.model_parm[x] = self.model_parm[x].float()
                    else:
                        flags_w = ('_w1' not in x) and ('_w2' not in x)
                        flagsw = ('w1' not in x) and ('w2' not in x)
                        flagsa = ('a1' not in x) and ('b2' not in x)
                        flagsg = ('g1' not in x) and ('g2' not in x)

                        if (len(self.model_parm[x].shape) == 2) and ('emb' not in x) and flags_w and flagsw and flagsa and flagsg and ('r_k' not in x):
                            if WTYPE != torch.uint8:
                                self.model_parm[x] = self.model_parm[x].to(dtype=WTYPE)
                            else:
                                self.model_parm[x] = self.model_parm[x].float()

                                if self.model_parm[x].shape[0] > self.model_parm[x].shape[1]:
                                    self.model_parm[x+'_my'] = torch.amin(self.model_parm[x], dim=1).unsqueeze(1)
                                    self.model_parm[x] = self.model_parm[x] - self.model_parm[x+'_my']
                                    self.model_parm[x+'_mx'] = torch.amin(self.model_parm[x], dim=0)
                                    self.model_parm[x] = self.model_parm[x] - self.model_parm[x+'_mx']
                                    self.model_parm[x+'_rx'] = torch.amax(self.model_parm[x], dim=0)
                                    self.model_parm[x] = self.model_parm[x] / self.model_parm[x+'_rx']
                                    self.model_parm[x+'_ry'] = torch.amax(self.model_parm[x], dim=1).unsqueeze(1)
                                    self.model_parm[x] = self.model_parm[x] / self.model_parm[x+'_ry']
                                else:
                                    self.model_parm[x+'_mx'] = torch.amin(self.model_parm[x], dim=0)
                                    self.model_parm[x] = self.model_parm[x] - self.model_parm[x+'_mx']
                                    self.model_parm[x+'_my'] = torch.amin(self.model_parm[x], dim=1).unsqueeze(1)
                                    self.model_parm[x] = self.model_parm[x] - self.model_parm[x+'_my']
                                    self.model_parm[x+'_rx'] = torch.amax(self.model_parm[x], dim=0)
                                    self.model_parm[x] = self.model_parm[x] / self.model_parm[x+'_rx']
                                    self.model_parm[x+'_ry'] = torch.amax(self.model_parm[x], dim=1).unsqueeze(1)
                                    self.model_parm[x] = self.model_parm[x] / self.model_parm[x+'_ry']

                                self.model_parm[x] = torch.clip(torch.floor(self.model_parm[x] * 256), min=0, max=255).to(dtype=torch.uint8)
                                self.model_parm[x+'_mx'] = self.model_parm[x+'_mx'].to(dtype=ATYPE).contiguous()
                                self.model_parm[x+'_rx'] = (self.model_parm[x+'_rx'] / 16).to(dtype=ATYPE).contiguous()
                                self.model_parm[x+'_my'] = self.model_parm[x+'_my'].to(dtype=ATYPE).contiguous()
                                self.model_parm[x+'_ry'] = (self.model_parm[x+'_ry'] / 16).to(dtype=ATYPE).contiguous()
                        else:
                            self.model_parm[x] = self.model_parm[x].to(dtype=ATYPE)
                            if self.version >= 7.0:
                                self.model_parm[x] = torch.squeeze(self.model_parm[x])
                
                if convert_and_save_and_exit == None:
                    # Load Model parm to RAM
                    if 'emb.' in x:
                        self.model_parm[x] = self.model_parm[x].contiguous()
                    elif (['is_tmp_layer']) and (x.endswith('key.weight') or x.endswith('value.weight') or x.endswith('receptance.weight') or x.endswith('output.weight')):
                        try:
                            self.model_parm[x] = self.model_parm[x].contiguous().pin_memory() # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
                        except:
                            print('Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower.')
                    elif DEVICE != 'cpu':
                        self.model_parm[x] = self.model_parm[x].to(device=DEVICE).contiguous()
                    
                    if (['is_tmp_layer']) or (DEVICE != 'cpu'):
                        try:
                            self.model_parm[x+'_mx'] = self.model_parm[x+'_mx'].to(device=DEVICE).contiguous()
                            self.model_parm[x+'_rx'] = self.model_parm[x+'_rx'].to(device=DEVICE).contiguous()
                            self.model_parm[x+'_my'] = self.model_parm[x+'_my'].to(device=DEVICE).contiguous()
                            self.model_parm[x+'_ry'] = self.model_parm[x+'_ry'].to(device=DEVICE).contiguous()
                        except:
                            pass

                if 'ffn.value.weight' in x:
                    gc.collect()
                    if 'cuda' in self.strategy_string:
                        torch.cuda.empty_cache()

                shape = [i for i in self.model_parm[x].shape if i != 1]
                if len(shape) > 2:
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)} {str(shape[2]).rjust(5)}"
                elif len(shape) > 1:
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}      "
                else:
                    shape = f" {str(shape[0]).rjust(5)}            "
                if layer_id == 0 or layer_id >= self.n_layer-1:
                    if print_need_newline:
                        verbose_out('\n', end = '')
                        print_need_newline = False
                    wtype = str(self.model_parm[x].dtype).replace('torch.', '')
                    wtype = wtype.replace('float32', 'f32').replace('bfloat16', 'bf16').replace('float16', 'f16').replace('uint8', 'i8')
                    verbose_out(x.ljust(32), wtype.rjust(4), str(self.model_parm[x].device).rjust(8), shape, ' (pinned)' if self.model_parm[x].is_pinned() else '')
                else:
                    print_need_newline = True
                    verbose_out('\r.' if layer_id % 10 == 0 else '.', end = '', flush = True)
            

            if convert_and_save_and_exit:
                self.model_parm['_strategy'] = self.strategy_string
                self.model_parm['_rescale_layer'] = self.rescale_layer
                # self.model_parm['_version'] = '0.7'
                if not convert_and_save_and_exit.endswith('.pth'):
                    convert_and_save_and_exit += '.pth'
                verbose_out(f'Saving to {convert_and_save_and_exit}...')
                torch.save(self.model_parm, convert_and_save_and_exit)
                verbose_out(f'Converted and saved. Now this will exit.')
                exit(0)
            
            if self.version == 4.0:
                from .model_src.RWKV_v4 import RWKV_v4
                self.rwkv4_model = RWKV_v4(self.model_parm, self.strategy, self.n_layer, self.rescale_layer)

            elif self.version == 5.2:
                from .model_src.RWKV_v5 import RWKV_x050
                self.rwkv5_model = RWKV_x050(self.model_parm, self.strategy, self.n_layer, self.rescale_layer)

            elif self.version == 6.0: # and os.environ["RWKV_CUDA_ON"] == '1':
                from .model_src.RWKV_v6 import RWKV_x060
                self.rwkv6_model = RWKV_x060(self.model_parm, self.strategy, self.n_layer, self.rescale_layer, self.time_state)

            elif self.version >= 7.0:
                from .model_src.RWKV_v7 import RWKV_x070
                self.rwkv7_model = RWKV_x070(self.model_parm, self.strategy, self.n_layer, self.n_head, self.rescale_layer)
            else:
                exit(f"Your model(Version {self.version:.1f}) is not supported.")


            gc.collect()
            if 'cuda' in self.strategy_string:
                torch.cuda.empty_cache()


    def forward(self, tokens, state, full_output=False):
        if self.version >= 7.0:
            return self.rwkv7_model(tokens, state, full_output)
        elif self.version >= 6.0:
            return self.rwkv6_model(tokens, state, full_output)
        elif self.version >= 5.0:
            return self.rwkv5_model(tokens, state, full_output)
        else:
            return self.rwkv4_model(tokens, state, full_output)
