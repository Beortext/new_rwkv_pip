print(f'\n### RWKV-7 "Goose" enabled ###\n')

import torch, os
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

if os.environ.get('RWKV_JIT_ON') == '1':
    uesModule = torch.jit.ScriptModule
    useFunction = torch.jit.script_method
    useStatic = torch.jit.script
else:
    uesModule = torch.nn.Module
    useFunction = lambda x: x
    useStatic = lambda x: x

from typing import List

HEAD_SIZE = 64

current_path = os.path.dirname(os.path.abspath(__file__))

from ..matmul_op import matmul


########################################################################################################

class RWKV_x070(uesModule):
    def __init__(self, z, strategy, n_layer, n_head, rescale_layer):
        global HEAD_SIZE
        super().__init__()
        self.eval()
        self.model_parm = z
        self.n_layer = n_layer
        self.strategy = strategy
        self.rescale_layer = rescale_layer
        self.n_head = n_head
        self.head_size = int(z['blocks.0.att.r_k'].shape[0] // n_head)
        HEAD_SIZE = self.head_size
        self.vocab_size, self.n_embd = z['emb.weight'].shape

    def forward(self, tokens, states, full_output=False):
        if states == None:
            states = [None for _ in range(self.n_layer * 3)]
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                
                states[i*3+0] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                states[i*3+1] = torch.zeros((self.n_head, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device=dev).contiguous()
                states[i*3+2] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

        if type(tokens) is list:
            if len(tokens) > 1:
                return self.forward_seq(tokens, states, full_output)
            else:
                return self.forward_one(tokens[0], states)
        else:
            return self.forward_one(tokens, states)

    # @MyFunction
    def forward_one(self, token:int, states:List[torch.Tensor]):
        with torch.no_grad(): 
            model_parm = self.model_parm
            x = model_parm['emb.weight'][token]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                block_id = f'blocks.{i}.'
                tmix_layer_id = f'blocks.{i}.att.'
                cmix_layer_id = f'blocks.{i}.ffn.'

                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                wtype = dds.wtype

                x = x.to(dtype=atype, device=dev)

                rw = model_parm[f"{tmix_layer_id}receptance.weight"]
                kw = model_parm[f"{tmix_layer_id}key.weight"]
                vw = model_parm[f"{tmix_layer_id}value.weight"]
                ow = model_parm[f"{tmix_layer_id}output.weight"]

                if dds.is_tmp_layer:
                    rw = rw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)

                rmx = model_parm[f'{tmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{tmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{tmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{tmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                kmx = model_parm[f'{tmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{tmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{tmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{tmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{tmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{tmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{tmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{tmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                xx = F.layer_norm(x, (self.n_embd,), weight=model_parm[f"{block_id}ln1.weight"], bias=model_parm[f"{block_id}ln1.bias"])

                xx, states[i*3+0], states[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, 
                    xx, states[i*3+0], v_first, states[i*3+1],
                    model_parm[f"{tmix_layer_id}ln_x.weight"], model_parm[f"{tmix_layer_id}ln_x.bias"],
                    model_parm[f"{tmix_layer_id}x_r"], model_parm[f"{tmix_layer_id}x_w"], model_parm[f"{tmix_layer_id}x_k"], 
                    model_parm[f"{tmix_layer_id}x_v"], model_parm[f"{tmix_layer_id}x_a"], model_parm[f"{tmix_layer_id}x_g"],
                    model_parm[f"{tmix_layer_id}a0"], model_parm[f"{tmix_layer_id}a1"], model_parm[f"{tmix_layer_id}a2"], 
                    model_parm[f"{tmix_layer_id}v0"], model_parm[f"{tmix_layer_id}v1"], model_parm[f"{tmix_layer_id}v2"],
                    model_parm[f"{tmix_layer_id}g1"], model_parm[f"{tmix_layer_id}g2"], 
                    model_parm[f"{tmix_layer_id}w0"], model_parm[f"{tmix_layer_id}w1"], model_parm[f"{tmix_layer_id}w2"], 
                    model_parm[f"{tmix_layer_id}k_k"], model_parm[f"{tmix_layer_id}k_a"], model_parm[f"{tmix_layer_id}r_k"],
                    rw, kw, vw, ow,
                    rmx, rrx, rmy, rry,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    omx, orx, omy, ory
                    )
                x = x + xx

                if dds.is_tmp_layer:
                    del rw, kw, vw, ow

                kw = model_parm[f'{cmix_layer_id}key.weight']
                vw = model_parm[f'{cmix_layer_id}value.weight']

                if dds.is_tmp_layer:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{cmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{cmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{cmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{cmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{cmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{cmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{cmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{cmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                xx = F.layer_norm(x, (self.n_embd,), weight=model_parm[f"{block_id}ln2.weight"], bias=model_parm[f"{block_id}ln2.bias"])

                xx, states[i*3+2] = RWKV_x070_CMix_one(
                    xx, states[i*3+2], model_parm[f"{cmix_layer_id}x_k"], 
                    kw, vw,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry
                    )
                x = x + xx
            
                # if math.isnan(torch.min(x).item()): print(idx, i)
                if dds.is_tmp_layer:               
                    del kw, vw, rw

                if self.rescale_layer > 0:
                    if (i+1) % self.rescale_layer == 0:
                        x = x / 2

            dds = self.strategy[self.n_layer]
            x = x.to(dtype=dds.atype, device=dds.device)

            x = F.layer_norm(x, (self.n_embd,), weight=model_parm['ln_out.weight'], bias=model_parm['ln_out.bias'])
            if model_parm['head.weight'].dtype != torch.uint8:
                x = x @ model_parm['head.weight']
            else:
                x = matmul(x, model_parm['head.weight'], model_parm['head.weight_mx'], model_parm['head.weight_rx'], model_parm['head.weight_my'], model_parm['head.weight_ry'])

            return x.float(), states
        
    # @MyFunction
    def forward_seq(self, tokens:List[int], states:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            model_parm = self.model_parm
            x = model_parm['emb.weight'][tokens]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                block_id = f'blocks.{i}.'
                tmix_layer_id = f'blocks.{i}.att.'
                cmix_layer_id = f'blocks.{i}.ffn.'

                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                wtype = dds.wtype

                x = x.to(dtype=atype, device=dev)

                rw = model_parm[f"{tmix_layer_id}receptance.weight"]
                kw = model_parm[f"{tmix_layer_id}key.weight"]
                vw = model_parm[f"{tmix_layer_id}value.weight"]
                ow = model_parm[f"{tmix_layer_id}output.weight"]

                if dds.is_tmp_layer:
                    rw = rw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)

                rmx = model_parm[f'{tmix_layer_id}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = model_parm[f'{tmix_layer_id}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = model_parm[f'{tmix_layer_id}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = model_parm[f'{tmix_layer_id}receptance.weight_ry'] if wtype == torch.uint8 else x

                kmx = model_parm[f'{tmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{tmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{tmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{tmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{tmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{tmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{tmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{tmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                xx = F.layer_norm(x, (self.n_embd,), weight=model_parm[f"{block_id}ln1.weight"], bias=model_parm[f"{block_id}ln1.bias"])

                xx, states[i*3+0], states[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, 
                    xx, states[i*3+0], v_first, states[i*3+1],
                    model_parm[f"{tmix_layer_id}ln_x.weight"], model_parm[f"{tmix_layer_id}ln_x.bias"],
                    model_parm[f"{tmix_layer_id}x_r"], model_parm[f"{tmix_layer_id}x_w"], model_parm[f"{tmix_layer_id}x_k"], 
                    model_parm[f"{tmix_layer_id}x_v"], model_parm[f"{tmix_layer_id}x_a"], model_parm[f"{tmix_layer_id}x_g"],
                    model_parm[f"{tmix_layer_id}a0"], model_parm[f"{tmix_layer_id}a1"], model_parm[f"{tmix_layer_id}a2"], 
                    model_parm[f"{tmix_layer_id}v0"], model_parm[f"{tmix_layer_id}v1"], model_parm[f"{tmix_layer_id}v2"],
                    model_parm[f"{tmix_layer_id}g1"], model_parm[f"{tmix_layer_id}g2"], 
                    model_parm[f"{tmix_layer_id}w0"], model_parm[f"{tmix_layer_id}w1"], model_parm[f"{tmix_layer_id}w2"], 
                    model_parm[f"{tmix_layer_id}k_k"], model_parm[f"{tmix_layer_id}k_a"], model_parm[f"{tmix_layer_id}r_k"],
                    rw, kw, vw, ow,
                    rmx, rrx, rmy, rry,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    omx, orx, omy, ory
                    )
                x = x + xx

                if dds.is_tmp_layer:
                    del rw, kw, vw, ow

                kw = model_parm[f'{cmix_layer_id}key.weight']
                vw = model_parm[f'{cmix_layer_id}value.weight']

                if dds.is_tmp_layer:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)

                kmx = model_parm[f'{cmix_layer_id}key.weight_mx'] if wtype == torch.uint8 else x
                krx = model_parm[f'{cmix_layer_id}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = model_parm[f'{cmix_layer_id}key.weight_my'] if wtype == torch.uint8 else x
                kry = model_parm[f'{cmix_layer_id}key.weight_ry'] if wtype == torch.uint8 else x

                vmx = model_parm[f'{cmix_layer_id}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = model_parm[f'{cmix_layer_id}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = model_parm[f'{cmix_layer_id}value.weight_my'] if wtype == torch.uint8 else x
                vry = model_parm[f'{cmix_layer_id}value.weight_ry'] if wtype == torch.uint8 else x

                xx = F.layer_norm(x, (self.n_embd,), weight=model_parm[f"{block_id}ln2.weight"], bias=model_parm[f"{block_id}ln2.bias"])

                xx, states[i*3+2] = RWKV_x070_CMix_seq(
                    xx, states[i*3+2], model_parm[f"{cmix_layer_id}x_k"], 
                    kw, vw,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry
                    )
                x = x + xx

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
                x = matmul(x, model_parm['head.weight'], model_parm['head.weight_mx'], model_parm['head.weight_rx'], model_parm['head.weight_my'], model_parm['head.weight_ry'])

            return x.float(), states

########################################################################################################

@useStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x_ln, x_prev, v_first, state, ln_w, ln_b, x_r, x_w, x_k, x_v, x_a, x_g, a0, a1, a2, v0, v1, v2, g1, g2, w0, w1, w2, k_k, k_a, r_k, rw, kw, vw, ow, rmx, rrx, rmy, rry, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, omx, orx, omy, ory):
    xx = x_prev - x_ln
    xr, xw, xk, xv, xa, xg = x_ln+xx*x_r, x_ln+xx*x_w, x_ln+xx*x_k, x_ln+xx*x_v, x_ln+xx*x_a, x_ln+xx*x_g

    r = matmul(xr, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32) #xr @ R_
    k = matmul(xk, kw, kmx, krx, kmy, kry, output_dtype=torch.float32) #xk @ K_
    v = matmul(xv, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32) #xv @ V_

    w = torch.tanh(xw @ w1) @ w2
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state @ r.view(H,N,1))

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    out = xx.to(dtype=x_ln.dtype) * g

    return matmul(out, ow, omx, orx, omy, ory), x_ln, state, v_first

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    load(name="wkv7s", sources=[f"{current_path}/../cuda/rwkv7_op.cpp", f"{current_path}/../cuda/rwkv7.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                T, C = r.size()
                H = C // HEAD_SIZE

                assert HEAD_SIZE == C // H
                # assert all(x.dtype == torch.float32 for x in [w, state])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])

                y = torch.empty((T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)

                if r.dtype == torch.float16:
                    torch.ops.wkv7s.forward_fp16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif r.dtype == torch.bfloat16:
                    torch.ops.wkv7s.forward_bf16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif r.dtype == torch.float32:
                    torch.ops.wkv7s.forward_fp32(1, T, C, H, state, r, w, k, v, a, b, y)

                return y

    def RWKV7_OP(state, r, w, k, v, a, b):
        return WKV_7.apply(state, r, w, k, v, a, b)
    
    @useStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x_ln, x_prev, v_first, state, ln_w, ln_b, x_r, x_w, x_k, x_v, x_a, x_g, a0, a1, a2, v0, v1, v2, g1, g2, w0, w1, w2, k_k, k_a, r_k, rw, kw, vw, ow, rmx, rrx, rmy, rry, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, omx, orx, omy, ory):
        T = x_ln.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x_ln[:-1,:])) - x_ln
        xr, xw, xk, xv, xa, xg = x_ln+xx*x_r, x_ln+xx*x_w, x_ln+xx*x_k, x_ln+xx*x_v, x_ln+xx*x_a, x_ln+xx*x_g

        r = matmul(xr, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32) #xr @ R_
        k = matmul(xk, kw, kmx, krx, kmy, kry, output_dtype=torch.float32) #xk @ K_
        v = matmul(xv, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32) #xv @ V_

        w = torch.tanh(xw @ w1) @ w2
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = -F.softplus(-(w0 + w)) - 0.5
        xx = RWKV7_OP(state, r, w.float(), k, v, -kk, kk*a.float())

        xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        out = xx.to(dtype=x_ln.dtype) * g

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], state, v_first
else:
    @useStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x_ln, x_prev, v_first, state, ln_w, ln_b, x_r, x_w, x_k, x_v, x_a, x_g, a0, a1, a2, v0, v1, v2, g1, g2, w0, w1, w2, k_k, k_a, r_k, rw, kw, vw, ow, rmx, rrx, rmy, rry, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, omx, orx, omy, ory):
        T = x_ln.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x_ln[:-1,:])) - x_ln
        xr, xw, xk, xv, xa, xg = x_ln+xx*x_r, x_ln+xx*x_w, x_ln+xx*x_k, x_ln+xx*x_v, x_ln+xx*x_a, x_ln+xx*x_g

        r = matmul(xr, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32) #xr @ R_
        k = matmul(xk, kw, kmx, krx, kmy, kry, output_dtype=torch.float32) #xk @ K_
        v = matmul(xv, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32) #xv @ V_

        w = torch.tanh(xw @ w1) @ w2
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
            vk = v_.view(H,N,1) @ k_.view(H,1,N)
            ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
            state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
            xx[t] = (state.to(dtype=x_ln.dtype) @ r_.view(H,N,1)).view(H*N)


        xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        out = xx.to(dtype=x_ln.dtype) * g

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], state, v_first

########################################################################################################

@useStatic
def RWKV_x070_CMix_one(x_ln, x_prev, x_k, kw, vw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry):
    xx = x_prev - x_ln
    kx = x_ln + xx * x_k
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return matmul(vx, vw, vmx, vrx, vmy, vry), x_ln

@useStatic
def RWKV_x070_CMix_seq(x_ln, x_prev, x_k, kw, vw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry):
    xx = torch.cat((x_prev.unsqueeze(0), x_ln[:-1,:])) - x_ln
    kx = x_ln + xx * x_k
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return matmul(vx, vw, vmx, vrx, vmy, vry), x_ln[-1,:]