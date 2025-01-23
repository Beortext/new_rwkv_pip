# new_rwkv_pip

[English docment](readme_en.md)

## 介绍

重构自 https://github.com/BlinkDL/ChatRWKV 的 `rwkv_pip_package`，支持 _rwkv v4/v5.2/v6/v7_ 的 **world** 系列 **LLM** 模型，并进行了一系列改进以满足更多应用需求。

依赖于 `torch` 框架。（如果需要用到 **DirectML**，还需要 `torch_directml` 包）

`utils.py` 中封装有推理管线 `Pipeline`，集成有 **state** 加载（rwkv-v6 独有功能）、预填充、生成（续写）和输出采样功能，**world** 模型的 **Tokenizer** 也有集成。同时，为方便用户快速上手，还提供了一份完整的聊天系统演示：`Chat_Demo.py`。

## 使用方法

### 1. 设置环境，调整运行方式

在运行前，需先设置环境变量，以启用不同的优化功能：

```python
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
os.environ["RWKV_DML_ON"] = "1"
```

通过设置 `RWKV_JIT_ON = 1`, `RWKV_CUDA_ON = 1` 和 `RWKV_DML_ON = 1`，可以分别启用 **PyTorch JIT** 优化、**RWKV CUDA** 算子以及 **DirectML** 支持，从而提升模型的运行效率。

### 2. 加载模型和推理管线

模型加载非常便捷，只需以下几行代码：

```python
model = RWKV(model_path=MODEL_PATH, strategy=strategy)
pipline = Pipline(model)
```

**strategy** 的设置可以参照下表，与 ChatRWKV 的 **strategy** 参数完全一致。不同策略适配不同硬件环境与性能需求：

| 策略                                | VRAM & RAM                          | 速度说明                                                                   |
| ----------------------------------- | ----------------------------------- | -------------------------------------------------------------------------- |
| **cpu fp32**                        | 7B 模型需要 32GB 内存               | 适合 Intel，AMD 性能极慢（PyTorch CPU gemv 问题，仅单核运行）。            |
| **cpu bf16**                        | 7B 模型需要 16GB 内存               | 支持 bfloat16 的新 Intel CPU（如 Xeon Platinum）速度较快。                 |
| **cpu fp32i8**                      | 7B 模型需要 12GB 内存               | 速度较慢（比 `cpu fp32` 更慢）。                                           |
| **cuda fp16**                       | 7B 模型需要 15GB VRAM               | 速度最快，但 VRAM 需求高。                                                 |
| **cuda fp16i8**                     | 7B 模型需要 9GB VRAM                | 速度较快；设置 `os.environ["RWKV_CUDA_ON"] = '1'` 可减少 1~2GB VRAM。      |
| **cuda fp16i8 \*20 -> cuda fp16**   | VRAM 介于 fp16 和 fp16i8 之间       | 前 20 层量化，其余层用 fp16。根据剩余 VRAM 调整量化层数。                  |
| **cuda fp16i8 \*20+**               | 比 fp16i8 更少 VRAM                 | 前 20 层固定量化，其他层动态加载（未固定层慢 3 倍）。根据 VRAM 增减层数。  |
| **cuda fp16i8 \*20 -> cpu fp32**    | 比 fp16i8 更少 VRAM，但内存占用更高 | 前 20 层量化固定到 GPU，其余层用 CPU fp32。CPU 强时更快，按需调整层数。    |
| **cuda:0 fp16 \*20 -> cuda:1 fp16** | 双卡分配                            | 卡 1（cuda:0）运行前 20 层，卡 2（cuda:1）运行剩余层。可混合 fp16/fp16i8。 |

#### 注意事项

目前只有 **RWKV-v7** 可以正确支持量化功能，这是一项已知 BUG（但可能不会修复）。如果需要量化功能，请优先考虑 rwkv-pip。

### 3. 使用推理管线

推理管线功能强大且易于使用。以下是预填充和生成示例：

```python
states = pipline.prefill(
    "User: hi\n\nAssistant: Hello, I am your AI Assistant. Are your question?\n\n",
    states=None
)
states = pipline.generate(
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

通过 `pipline.prefill()`，用户可以将 Prompt 预嵌入到 **state** 中，减少每次生成时的初始化开销。而 `pipline.generate()` 是模型生成函数，支持按需调整生成长度、温度、采样参数等。

#### 参数说明

- **states**: 模型状态参数，保存对话上下文。
- **temperature**: 控制生成文本的随机性。
- **top_p** 和 **top_k**: 限定采样的概率和词表大小。
- **presence** 和 **frequency**: 控制重复惩罚。
- **penalty_decay**: 惩罚权重的衰减系数。

通过合理设置这些参数，可以显著优化生成结果，满足不同任务需求。

此外，`Chat_Demo.py` 提供了一个完整的聊天系统演示，用户可以参考其中代码更快地搭建自己的应用场景。
