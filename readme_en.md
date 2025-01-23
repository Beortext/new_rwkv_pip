# new_rwkv_pip

[中文文档](README.md)

## Introduction

This is a restructured version of `rwkv_pip_package` from https://github.com/BlinkDL/ChatRWKV, supporting **world** series **LLM** models for _rwkv v4/v5.2/v6/v7_. Significant improvements have been made to meet broader application requirements.

The package depends on the `torch` framework. (For **DirectML** usage, the `torch_directml` package is also required.)

The file `utils.py` contains a prebuilt inference pipeline `Pipeline`, which integrates features such as **state** loading (exclusive to rwkv-v6), pre-filling, text generation, and output sampling. It also includes the **Tokenizer** for **world** models. A complete chatbot system demo is provided in `Chat_Demo.py` to help users get started quickly.

## How to Use

### 1. Set Environment and Adjust Execution

Before running, set the environment variables to enable various optimization features:

```python
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
os.environ["RWKV_DML_ON"] = "1"
```

Setting `RWKV_JIT_ON = 1`, `RWKV_CUDA_ON = 1`, and `RWKV_DML_ON = 1` enables **PyTorch JIT** optimization, **RWKV CUDA** operators, and **DirectML** support, respectively, to enhance model performance.

### 2. Load Model and Inference Pipeline

Loading the model is straightforward with just a few lines of code:

```python
model = RWKV(model_path=MODEL_PATH, strategy=strategy)
pipline = Pipline(model)
```

The **strategy** parameter is consistent with ChatRWKV and can be configured according to the table below to match different hardware environments and performance needs:

| Strategy                            | VRAM & RAM                          | Performance Details                                                        |
| ----------------------------------- | ----------------------------------- | -------------------------------------------------------------------------- |
| **cpu fp32**                        | 32GB RAM for a 7B model             | Suitable for Intel, AMD; very slow (PyTorch CPU gemv issue, single-threaded only). |
| **cpu bf16**                        | 16GB RAM for a 7B model             | Faster on newer Intel CPUs supporting bfloat16 (e.g., Xeon Platinum).      |
| **cpu fp32i8**                      | 12GB RAM for a 7B model             | Slower than `cpu fp32`.                                                    |
| **cuda fp16**                       | 15GB VRAM for a 7B model            | Fastest speed, but high VRAM requirement.                                  |
| **cuda fp16i8**                     | 9GB VRAM for a 7B model             | Quite fast; set `os.environ["RWKV_CUDA_ON"] = '1'` to save 1–2GB VRAM.    |
| **cuda fp16i8 \*20 -> cuda fp16**   | VRAM between fp16 and fp16i8        | First 20 layers quantized, rest use fp16. Adjust quantized layers as needed. |
| **cuda fp16i8 \*20+**               | Less VRAM than fp16i8               | First 20 layers fixed quantized, others dynamically loaded (3x slower for non-fixed layers). |
| **cuda fp16i8 \*20 -> cpu fp32**    | Less VRAM than fp16i8, higher RAM   | First 20 layers fixed on GPU, rest run on CPU fp32. Faster with strong CPUs. Adjust layers as needed. |
| **cuda:0 fp16 \*20 -> cuda:1 fp16** | Dual GPU allocation                 | First 20 layers run on GPU 0 (cuda:0), remaining layers on GPU 1 (cuda:1). Can mix fp16/fp16i8. |

#### Notes

Currently, only **RWKV-v7** correctly supports quantization. This is a known issue (and may not be fixed). For quantization, consider using rwkv-pip.

### 3. Use the Inference Pipeline

The inference pipeline is powerful and easy to use. Below are examples for pre-filling and text generation:

```python
states = pipeline.prefill(
    "User: hi\n\nAssistant: Hello, I am your AI Assistant. Are your question?\n\n",
    states=None
)
states = pipeline.generate(
    "User: hi\n\nAssistant:",
    gen_len=4096,
    states=None,
    temperature=1.0,
    top_p=0.3,
    top_k=0,
    presence=0.5,
    frequency=0.5,
    penalty_decay=0.996
)
```

Through `pipeline.prefill()`, users can pre-embed the Prompt into the state, reducing the initialization overhead for each generation. Meanwhile, `pipeline.generate()` is the model generation function, which supports on-demand adjustments of generation length, temperature, sampling parameters, and more.

**Parameter Description**

- **states**: Model state parameters that save the conversation context.
- **temperature**: Controls the randomness of the generated text.
- **top_p** and **top_k**: Limit the sampling probability and vocabulary size.
- **presence** and **frequency**: Control repetition penalties.
- **penalty_decay**: The decay coefficient for penalty weights.

By setting these parameters appropriately, the generation results can be significantly optimized to meet the needs of different tasks.

Additionally, `Chat_Demo.py` provides a complete demonstration of a chat system. Users can refer to the code within to quickly build their own application scenarios.