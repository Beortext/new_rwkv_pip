print(f'\n### RWKV-5 "Eagle" enabled ###\n')

import torch, os
from typing import List
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

if os.environ.get('RWKV_JIT_ON') == '1':
    uesModule = torch.jit.ScriptModule
    useFunction = torch.jit.script_method
    useStatic = torch.jit.script
else:
    uesModule = torch.nn.Module
    useFunction = lambda x: x
    useStatic = lambda x: x

current_path = os.path.dirname(os.path.abspath(__file__))

HEAD_SIZE = 64

from ..matmul_op import matmul


class RWKV_x050(uesModule):
    def __init__(self, model_parm, strategy, n_layer, rescale_layer):
        global HEAD_SIZE
        super().__init__()
        self.eval()
        self.model_parm = model_parm
        self.n_layer = n_layer
        self.strategy = strategy
        self.rescale_layer = rescale_layer
        self.n_head, self.head_size, _ = model_parm['blocks.0.att.time_first'].shape
        HEAD_SIZE = self.head_size
        self.n_embd = model_parm['emb.weight'].shape[-1]

    def forward(self, tokens, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.n_layer * 3)]
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                
                state[i*3+0] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                state[i*3+1] = torch.zeros((self.n_head, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i*3+2] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

        if type(tokens) is list:
            if len(tokens) > 1:
                return self.forward_seq(tokens, state, full_output)
            else:
                return self.forward_one(tokens[0], state)
        else:
            return self.forward_one(tokens, state)

    # @MyFunction
    def forward_one(self, token:int, states:List[torch.Tensor]):
        with torch.no_grad(): 
            model_parm = self.model_parm
            x = model_parm['emb.weight'][token]

            for i in range(self.n_layer):
                block_id = f'blocks.{i}.'
                tmix_layer_id = f'blocks.{i}.att.'
                cmix_layer_id = f'blocks.{i}.ffn.'

                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                wtype = dds.wtype

                x = x.to(dtype=atype, device=dev)

                # TimeMix Layrer Parm
                kw = model_parm[f'{tmix_layer_id}key.weight']
                vw = model_parm[f'{tmix_layer_id}value.weight']
                rw = model_parm[f'{tmix_layer_id}receptance.weight']
                gw = model_parm[f'{tmix_layer_id}gate.weight']
                ow = model_parm[f'{tmix_layer_id}output.weight']

                if dds.is_tmp_layer:
                    gw = gw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{tmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{tmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{tmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{tmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{tmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{tmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{tmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{tmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                rmx = model_parm[f'{tmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{tmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{tmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{tmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                gmx = model_parm[f'{tmix_layer_id}gate.weight_mx'] if wtype == torch.uint8 else x
                grx = model_parm[f'{tmix_layer_id}gate.weight_rx'] if wtype == torch.uint8 else x
                gmy = model_parm[f'{tmix_layer_id}gate.weight_my'] if wtype == torch.uint8 else x
                gry = model_parm[f'{tmix_layer_id}gate.weight_ry'] if wtype == torch.uint8 else x

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln1.weight'], bias=model_parm[block_id+'ln1.bias'])
                x_ln, states[i*3+0], states[i*3+1] = RWKV_x050_TMix_one(
                    x_ln, states[i*3+0], states[i*3+1],
                    model_parm[f'{tmix_layer_id}ln_x.weight'], model_parm[f'{tmix_layer_id}ln_x.bias'],
                    model_parm[f'{tmix_layer_id}time_mix_k'], model_parm[f'{tmix_layer_id}time_mix_v'], 
                    model_parm[f'{tmix_layer_id}time_mix_r'], model_parm[f'{tmix_layer_id}time_mix_g'],
                    model_parm[f'{tmix_layer_id}time_decay'], model_parm[f'{tmix_layer_id}time_first'],
                    kw, vw, rw, gw, ow,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    rmx, rrx, rmy, rry,
                    gmx, grx, gmy, gry,
                    omx, orx, omy, ory,
                )
                x = x + x_ln

                if dds.is_tmp_layer:
                    del kw, vw, rw, gw, ow

                # ChannelMix Layer Parm
                kw = model_parm[f'{cmix_layer_id}key.weight']
                vw = model_parm[f'{cmix_layer_id}value.weight']
                rw = model_parm[f'{cmix_layer_id}receptance.weight']

                if dds.is_tmp_layer:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{cmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{cmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{cmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{cmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{cmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{cmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{cmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{cmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                rmx = model_parm[f'{cmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{cmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{cmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{cmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln2.weight'], bias=model_parm[block_id+'ln2.bias'])
                x_ln, states[i*3+2] = RWKV_x050_CMix_one(
                        states[i*3+2], x_ln,
                        model_parm[f'{cmix_layer_id}time_mix_k'], model_parm[f'{cmix_layer_id}time_mix_r'],
                        kw, vw, rw,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,  
                    )
                x = x + x_ln

                if dds.is_tmp_layer:              
                    del kw, vw, rw

                if self.rescale_layer > 0:
                    if (i+1) % self.rescale_layer == 0:
                        x = x / 2
                # if math.isnan(torch.min(x).item()): print(idx, i)

            dds = self.strategy[self.n_layer]
            x = x.to(dtype=dds.atype, device=dds.device)

            x = F.layer_norm(x, (self.n_embd,), weight=model_parm['ln_out.weight'], bias=model_parm['ln_out.bias'])
            if model_parm['head.weight'].dtype != torch.uint8:
                    x = x @ model_parm['head.weight']
            else:
                model_parm = matmul(x, model_parm['head.weight'], model_parm['head.weight_mx'], model_parm['head.weight_rx'], model_parm['head.weight_my'], model_parm['head.weight_ry'])

            return x.float(), states
        
        
    # @MyFunction
    def forward_seq(self, tokens:List[int], states:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            model_parm = self.model_parm
            x = model_parm['emb.weight'][tokens]

            for i in range(self.n_layer):
                block_id = f'blocks.{i}.'
                tmix_layer_id = f'blocks.{i}.att.'
                cmix_layer_id = f'blocks.{i}.ffn.'

                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                wtype = dds.wtype

                x = x.to(dtype=atype, device=dev)

                # TimeMix Layrer Parm
                kw = model_parm[f'{tmix_layer_id}key.weight']
                vw = model_parm[f'{tmix_layer_id}value.weight']
                rw = model_parm[f'{tmix_layer_id}receptance.weight']
                gw = model_parm[f'{tmix_layer_id}gate.weight']
                ow = model_parm[f'{tmix_layer_id}output.weight']

                if dds.is_tmp_layer:
                    gw = gw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{tmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{tmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{tmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{tmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{tmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{tmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{tmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{tmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                rmx = model_parm[f'{tmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{tmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{tmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{tmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                gmx = model_parm[f'{tmix_layer_id}gate.weight_mx'] if wtype == torch.uint8 else x
                grx = model_parm[f'{tmix_layer_id}gate.weight_rx'] if wtype == torch.uint8 else x
                gmy = model_parm[f'{tmix_layer_id}gate.weight_my'] if wtype == torch.uint8 else x
                gry = model_parm[f'{tmix_layer_id}gate.weight_ry'] if wtype == torch.uint8 else x

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln1.weight'], bias=model_parm[block_id+'ln1.bias'])
                x_ln, states[i*3+0], states[i*3+1] = RWKV_x050_TMix_seq(
                    x_ln, states[i*3+0], states[i*3+1],
                    model_parm[f'{tmix_layer_id}ln_x.weight'], model_parm[f'{tmix_layer_id}ln_x.bias'],
                    model_parm[f'{tmix_layer_id}time_mix_k'], model_parm[f'{tmix_layer_id}time_mix_v'], 
                    model_parm[f'{tmix_layer_id}time_mix_r'], model_parm[f'{tmix_layer_id}time_mix_g'],
                    model_parm[f'{tmix_layer_id}time_decay'], model_parm[f'{tmix_layer_id}time_first'],
                    kw, vw, rw, gw, ow,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    rmx, rrx, rmy, rry,
                    gmx, grx, gmy, gry,
                    omx, orx, omy, ory,
                )
                x = x + x_ln

                if dds.is_tmp_layer:
                    del kw, vw, rw, gw, ow

                # ChannelMix Layer Parm
                kw = model_parm[f'{cmix_layer_id}key.weight']
                vw = model_parm[f'{cmix_layer_id}value.weight']
                rw = model_parm[f'{cmix_layer_id}receptance.weight']

                if dds.is_tmp_layer:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{cmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{cmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{cmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{cmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{cmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{cmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{cmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{cmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                rmx = model_parm[f'{cmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{cmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{cmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{cmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln2.weight'], bias=model_parm[block_id+'ln2.bias'])
                x_ln, states[i*3+2] = RWKV_x050_CMix_seq(
                        states[i*3+2], x_ln,
                        model_parm[f'{cmix_layer_id}time_mix_k'], model_parm[f'{cmix_layer_id}time_mix_r'],
                        kw, vw, rw,
                        kmx, krx, kmy, kry,
                        vmx, vrx, vmy, vry,
                        rmx, rrx, rmy, rry,  
                    )
                x = x + x_ln

                if dds.is_tmp_layer:               
                    del kw, vw, rw

                if self.rescale_layer > 0:
                    if (i+1) % self.rescale_layer == 0:
                        x = x / 2
            
            dds = self.strategy[self.n_layer]
            x = x[-1,:] if not full_output else x
            x = x.to(dtype=dds.atype, device=dds.device)

            x = F.layer_norm(x, (self.n_embd,), weight=model_parm['ln_out.weight'], bias=model_parm['ln_out.bias'])
            if model_parm['head.weight'].dtype != torch.uint8:
                    x = x @ model_parm['head.weight']
            else:
                model_parm = matmul(x, model_parm['head.weight'], model_parm['head.weight_mx'], model_parm['head.weight_rx'], model_parm['head.weight_my'], model_parm['head.weight_ry'])

            return x.float(), states


@useStatic
def RWKV_x050_TMix_one(x_ln, x_prv, state, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
    H = t_decay.shape[0]
    N = x_ln.shape[-1] // H
    
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    vx = x_ln * v_mix + x_prv * (1 - v_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)
    gx = x_ln * g_mix + x_prv * (1 - g_mix)

    r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
    g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

    a = matmul(k, v)
    out = r @ (t_first * a + state)
    state = a + t_decay * state

    out = out.flatten()
    out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
    out = out.to(dtype=x_ln.dtype) * g

    return matmul(out, ow, omx, orx, omy, ory), x_ln, state


if os.environ["RWKV_CUDA_ON"] == '1':
    from torch.utils.cpp_extension import load
    rwkv5 = load(name="rwkv5", sources=[f"{current_path}/../cuda/rwkv5_op.cpp", f"{current_path}/../cuda/rwkv5.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
        
    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, k, v, w, u):
            with torch.no_grad():
                T, C = r.size()
                H = C // HEAD_SIZE

                assert HEAD_SIZE == C // H
                assert state.dtype == torch.float32
                assert w.dtype == torch.float32
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()

                y = torch.empty((1, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
                
                if r.dtype == torch.bfloat16:
                    rwkv5.forward_bf16(1, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float16:
                    rwkv5.forward_fp16(1, T, C, H, state, r, k, v, w, u, y)
                elif r.dtype == torch.float32:
                    rwkv5.forward_fp32(1, T, C, H, state, r, k, v, w, u, y)
                
                return y
            
    def RWKV5_OP(state, r, k, v, w, u):
        return WKV_5.apply(state, r, k, v, w, u)
    
    @useStatic
    def RWKV_x050_TMix_seq(x_ln, x_prv, state, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        H = t_decay.shape[0]
        N = x_ln.shape[-1] // H
        T = x_ln.shape[0]

        x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
        kx = x_ln * k_mix + x_prv * (1 - k_mix)
        vx = x_ln * v_mix + x_prv * (1 - v_mix)
        rx = x_ln * r_mix + x_prv * (1 - r_mix)
        gx = x_ln * g_mix + x_prv * (1 - g_mix)

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        state = state.transpose(-1,-2).contiguous()
        out = RWKV5_OP(state, r, k, v, t_decay, t_first)
        state = state.transpose(-1,-2)

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x_ln.dtype) * g

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], state
            
else:
    @useStatic
    def RWKV_x050_TMix_seq(x_ln, x_prv, state, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        H = t_decay.shape[0]
        N = x_ln.shape[-1] // H
        T = x_ln.shape[0]

        x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
        kx = x_ln * k_mix + x_prv * (1 - k_mix)
        vx = x_ln * v_mix + x_prv * (1 - v_mix)
        rx = x_ln * r_mix + x_prv * (1 - r_mix)
        gx = x_ln * g_mix + x_prv * (1 - g_mix)

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + state)).squeeze(1)
            state = at + t_decay * state

        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x_ln.dtype) * g

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], state

@useStatic
def RWKV_x050_CMix_one(x_prv, x_ln, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)

    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return r * matmul(vx, vw, vmx, vrx, vmy, vry), x_ln

@useStatic
def RWKV_x050_CMix_seq(x_prv, x_ln, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)

    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return r * matmul(vx, vw, vmx, vrx, vmy, vry), x_ln[-1,:]