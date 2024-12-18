from typing import Optional
import os, torch

torch.backends.cuda.matmul.allow_tf32 = True

current_path = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('RWKV_JIT_ON') == '1':
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    MyFunction = lambda x: x
    MyStatic = lambda x: x

# ==================================[CUDA Kernel]===========================================

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load

    try:
        load(
            name=f"mm8_cuda",
            sources=[f"{current_path}/cuda/cuda_mm8_op.cpp", f"{current_path}/cuda/cuda_mm8.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
            verbose=True,
            extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            is_python_module=False)
        USE_CUBLAS_GEMM = True
    except:
        print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
        load(
            name=f"mm8_cuda",
            sources=[f"{current_path}/cuda/cuda_mm8_op.cpp", f"{current_path}/cuda/cuda_mm8.cu",],
            verbose=True,
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
            is_python_module=False)
        USE_CUBLAS_GEMM = False


    @MyStatic
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        torch.ops.mm8_cuda.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        return y
    @MyStatic
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (N,)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        torch.ops.mm8_cuda.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        return y.to(dtype=x.dtype)


    if USE_CUBLAS_GEMM:
        @MyStatic
        def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
            if output_dtype is None:
                output_dtype = a.dtype

            if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
                if len(a.shape) == 1:
                    assert len(b.shape) == 2
                    c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                    a = a.unsqueeze(0)

                else:
                    assert len(a.shape) == len(b.shape)
                    assert len(a.shape) == 2 or len(a.shape) == 3
                    # torch.empty((*a.shape[:-1], b.shape[-1])) doesn't work with jit
                    if len(a.shape) == 2:
                        c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
                    else:
                        c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
                        
                torch.ops.mm8_cuda.gemm_fp16_cublas(a, b, c)
                return c
            else:
                return (a @ b.to(device=a.device)).to(output_dtype)
    else:
        @MyStatic
        def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
            return (a @ b.to(device=a.device)).to(output_dtype)
else:
    @MyStatic
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        return (a @ b.to(device=a.device)).to(output_dtype)

# =============================================================================

@MyStatic
def torch_mm8(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def mm8_seq(x, w, mx, rx, my, ry):
    if w.device.type == 'cuda' and x.dtype == torch.float16:
        B, N, M = x.shape[0], w.shape[0], w.shape[1]
        return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
    else:
        return torch_mm8(x, w, mx, rx, my, ry)
@MyStatic
def mm8_one(x, w, mx, rx, my, ry):
    if w.device.type == 'cuda':
        N, M = w.shape[0], w.shape[1]
        return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
    else:
        return torch_mm8(x, w, mx, rx, my, ry)
    
@MyStatic
def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
    if len(x.shape) == 1:
        return mm8_one(x, w, mx, rx, my, ry)
    return mm8_seq(x, w, mx, rx, my, ry)

@MyStatic
def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")