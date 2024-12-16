print(f'\n### RWKV-4 "Dove" enabled ###\n')

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


from ..matmul_op import matmul


class RWKV_v4(uesModule):
    def __init__(self, model_parm, strategy, n_layer, rescale_layer):
        global HEAD_SIZE
        super().__init__()
        self.eval()
        self.model_parm = model_parm
        self.n_layer = n_layer
        self.strategy = strategy
        self.rescale_layer = rescale_layer
        self.n_att = model_parm['blocks.0.att.key.weight'].shape[0]
        self.n_embd = model_parm['emb.weight'].shape[-1]

    def forward(self, tokens, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.n_layer * 5)]
            for i in range(self.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                dds = self.strategy[i]
                dev = dds.device
                atype = dds.atype
                
                # dev = dd['device']
                # atype = dd['atype']
                state[i*5+0] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                state[i*5+1] = torch.zeros(self.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i*5+2] = torch.zeros(self.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                state[i*5+3] = torch.zeros(self.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
                state[i*5+4] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

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
                rw = model_parm[f'{tmix_layer_id}receptance.weight']
                kw = model_parm[f'{tmix_layer_id}key.weight']
                vw = model_parm[f'{tmix_layer_id}value.weight']
                ow = model_parm[f'{tmix_layer_id}output.weight']

                if dds.is_tmp_layer:
                    rw = rw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
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

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln1.weight'], bias=model_parm[block_id+'ln1.bias'])
                x_ln, states[i*5+0], states[i*5+1], states[i*5+2], states[i*5+3] = RWKV_v4_TMix_one(
                    x_ln, states[i*5+0], states[i*5+1], states[i*5+2], states[i*5+3],
                    model_parm[f'{tmix_layer_id}time_mix_k'], model_parm[f'{tmix_layer_id}time_mix_v'], model_parm[f'{tmix_layer_id}time_mix_r'], 
                    model_parm[f'{tmix_layer_id}time_decay'], model_parm[f'{tmix_layer_id}time_first'],
                    kw, vw, rw, ow,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    rmx, rrx, rmy, rry,
                    omx, orx, omy, ory,
                )
                x = x + x_ln

                if dds.is_tmp_layer:
                    del kw, vw, rw, ow

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
                x_ln, states[i*5+4] = RWKV_v4_CMix_one(
                        states[i*5+4], x_ln,
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
                rw = model_parm[f'{tmix_layer_id}receptance.weight']
                kw = model_parm[f'{tmix_layer_id}key.weight']
                vw = model_parm[f'{tmix_layer_id}value.weight']
                ow = model_parm[f'{tmix_layer_id}output.weight']

                if dds.is_tmp_layer:
                    rw = rw.to(device=dev, non_blocking=True)
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
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

                omx = model_parm[f'{tmix_layer_id}output.weight_mx'] if wtype == torch.uint8 else x
                orx = model_parm[f'{tmix_layer_id}output.weight_rx'] if wtype == torch.uint8 else x
                omy = model_parm[f'{tmix_layer_id}output.weight_my'] if wtype == torch.uint8 else x
                ory = model_parm[f'{tmix_layer_id}output.weight_ry'] if wtype == torch.uint8 else x

                x_ln = F.layer_norm(x, (x.shape[-1],), weight=model_parm[block_id+'ln1.weight'], bias=model_parm[block_id+'ln1.bias'])
                x_ln, states[i*5+0], states[i*5+1], states[i*5+2], states[i*5+3] = RWKV_v4_TMix_seq(
                    x_ln, states[i*5+0], states[i*5+1], states[i*5+2], states[i*5+3],
                    model_parm[f'{tmix_layer_id}time_mix_k'], model_parm[f'{tmix_layer_id}time_mix_v'], model_parm[f'{tmix_layer_id}time_mix_r'], 
                    model_parm[f'{tmix_layer_id}time_decay'], model_parm[f'{tmix_layer_id}time_first'],
                    kw, vw, rw, ow,
                    kmx, krx, kmy, kry,
                    vmx, vrx, vmy, vry,
                    rmx, rrx, rmy, rry,
                    omx, orx, omy, ory,
                )
                x = x + x_ln

                if dds.is_tmp_layer:
                    del kw, vw, rw, ow

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
                x_ln, states[i*5+4] = RWKV_v4_CMix_seq(
                        states[i*5+4], x_ln,
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
def RWKV_v4_TMix_one(x_ln, x_prv, aa, bb, pp, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    vx = x_ln * v_mix + x_prv * (1 - v_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)

    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

    ww = t_first + k
    p = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p)
    e2 = torch.exp(ww - p)

    wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2))
    out = r * wkv.to(dtype=x_ln.dtype)

    ww = t_decay + pp
    p = torch.maximum(ww, k)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(k - p)

    return matmul(out, ow, omx, orx, omy, ory), x_ln, aa, bb, pp


if os.environ["RWKV_CUDA_ON"] == '1':
    from torch.utils.cpp_extension import load
    rwkv4 = load(name=f"rwkv4", sources=[f"{current_path}/../cuda/rwkv4_op.cpp", f"{current_path}/../cuda/rwkv4.cu"], 
                                verbose=True, extra_cuda_cflags=["--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization"])
        
    class WKV_4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, u, k, v, aa, bb, pp):
            with torch.no_grad():
                T, C = k.size()

                assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
                assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()

                y = torch.empty((T, C), device=w.device, dtype=k.dtype, memory_format=torch.contiguous_format)
                
                rwkv4.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
                
                return y
            
    def RWKV4_OP(w, u, k, v, aa, bb, pp):
        return WKV_4.apply(w, u, k, v, aa, bb, pp)
    
    @useStatic
    def RWKV_v4_TMix_seq(x_ln, x_prv, aa, bb, pp, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
        kx = x_ln * k_mix + x_prv * (1 - k_mix)
        vx = x_ln * v_mix + x_prv * (1 - v_mix)
        rx = x_ln * r_mix + x_prv * (1 - r_mix)


        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        wkv = RWKV4_OP(t_decay, t_first, k, v, aa, bb, pp)
        out = r * wkv.to(dtype=x_ln.dtype)

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], aa, bb, pp
            
else:
    @useStatic
    def RWKV_v4_TMix_seq(x_ln, x_prv, aa, bb, pp, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        T = x_ln.shape[1]

        x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
        kx = x_ln * k_mix + x_prv * (1 - k_mix)
        vx = x_ln * v_mix + x_prv * (1 - v_mix)
        rx = x_ln * r_mix + x_prv * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        wkv = torch.empty_like(x_prv, dtype=k.dtype, device=k.device)
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            wkv[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2))
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        
        out = r * wkv.to(dtype=x_ln.dtype)

        return matmul(out, ow, omx, orx, omy, ory), x_ln[-1,:], aa, bb, pp

@useStatic
def RWKV_v4_CMix_one(x_prv, x_ln, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)

    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return r * matmul(vx, vw, vmx, vrx, vmy, vry), x_ln

@useStatic
def RWKV_v4_CMix_seq(x_prv, x_ln, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    x_prv = torch.cat((x_prv.unsqueeze(0), x_ln[:-1,:]))
    kx = x_ln * k_mix + x_prv * (1 - k_mix)
    rx = x_ln * r_mix + x_prv * (1 - r_mix)

    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2

    return r * matmul(vx, vw, vmx, vrx, vmy, vry), x_ln[-1,:]