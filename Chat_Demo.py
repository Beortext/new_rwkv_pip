########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print("RWKV Chat Simple Demo")

# 设置环境变量 | Set environment variables
import os

# 开启即时编译（提升性能）| Enable JIT compilation (improves performance)
os.environ["RWKV_JIT_ON"] = "1"  

# 开启CUDA加速（需要CUDA环境和编译器）| Enable CUDA acceleration (requires CUDA environment and compiler)
os.environ["RWKV_CUDA_ON"] = "1"  

# 指定CUDA架构（这里适配Tesla P40）| Specify CUDA architecture (here for Tesla P40)
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1"

# 导入模型和工具 | Import model and utilities
from src.model import RWKV  # 主模型类 | Main model class
from src.utils import Pipeline, chat_prompt  # 工具类 | Utility classes

########################################################################################################
# 模型配置参数 | Model configuration parameters
strategy = "cuda fp16i8"  # 运行策略：CUDA加速 + FP16混合精度 + int8量化（仅RWKV-7支持）
                         # Strategy: CUDA acceleration + FP16 mixed precision + int8 quantization (RWKV-7+ only)
CTX_LEN = 4096  # 上下文最大长度 | Maximum context length

# 模型文件路径 | Model file path
MODEL_NAME = f"/home/beortust/nfs_share/RwkvModelLib/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096"
# 初始状态文件（None表示使用默认空状态）| Initial state file (None uses vanilla zero state)
STATE_NAME = None  

########################################################################################################

# 加载模型 | Load model
print(f"Loading model - {MODEL_NAME}")
model = RWKV(model_path=MODEL_NAME, strategy=strategy)  # 创建模型实例 | Create model instance
pipeline = Pipeline(model)  # 创建处理管道 | Create processing pipeline

# 初始化对话状态 | Initialize conversation state
states = pipeline.prefill("User: hi\n\nAssistant: Hello, I'am your AI Assistant. Are your question?\n\n")

# 主对话循环 | Main conversation loop
while True:
    msg = chat_prompt()  # 获取用户输入 | Get user input (需要实现chat_prompt函数 | requires chat_prompt implementation)
    
    # 生成回复并更新状态 | Generate response and update state
    states = pipeline.generate(
        msg,        # 用户输入 | User input
        CTX_LEN,    # 上下文长度限制 | Context length limit
        states      # 历史对话状态 | Conversation history state
    )
