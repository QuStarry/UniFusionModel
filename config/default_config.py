"""
UniFusion 验证实验配置
核心验证目标：隐状态级模型调用是否比文本级 Tool Use 更有效
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    # 大模型（通用基座）
    large_model_name: str = "Qwen/Qwen2-7B-Instruct"
    large_model_dtype: str = "bfloat16"
    large_model_device: str = "cuda:0"
    # 在大模型的第几层提取/注入隐状态（支持负数，-1表示最后一层）
    large_hook_layer: int = -8  # 倒数第8层，通常是高级语义形成层

    # 专业小模型
    small_model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    small_model_dtype: str = "bfloat16"
    small_model_device: str = "cuda:1"
    # 小模型从第几层开始继续前向传播（前面的层被大模型替代）
    small_start_layer: int = 16  # 小模型前16层的计算被大模型+适配层替代
    # 小模型取第几层的输出作为增强隐状态
    small_output_layer: int = -1  # 最后一层

    # 是否加载小模型的完整权重（验证阶段设为True）
    load_small_full: bool = True


@dataclass
class AdapterConfig:
    """适配层配置"""
    # 适配层隐藏维度（通常设为小模型维度的1/2或1/4）
    hidden_dim: int = 2048
    # 适配层类型：linear / mlp / mlp_residual
    adapter_type: str = "mlp_residual"
    # Dropout
    dropout: float = 0.1
    # 是否使用层归一化
    use_layer_norm: bool = True
    # 初始化方式：xavier / identity / zero
    init_method: str = "identity"


@dataclass
class GatingConfig:
    """门控融合配置"""
    # 门控类型：scalar（标量）/ vector（逐维）/ input_dependent（输入相关）
    gating_type: str = "input_dependent"
    # 门控初始值（0=完全用大模型，1=完全用小模型增强）
    init_gate: float = 0.5
    # 门控温度
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """适配层校准训练配置"""
    # 训练数据
    train_data_path: str = "data/calibration_samples.jsonl"
    num_calibration_samples: int = 200  # 少样本校准，50-200条即可

    # 优化器
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    max_steps: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 2

    # 损失函数权重
    # 隐状态MSE损失权重
    hidden_mse_weight: float = 1.0
    # 输出KL散度损失权重（让融合后的输出分布接近小模型）
    output_kl_weight: float = 0.5
    # 大模型原生输出保持损失（防止门控完全偏向小模型）
    native_keep_weight: float = 0.1

    # 保存
    output_dir: str = "checkpoints/adapter_calibration"
    save_steps: int = 100
    logging_steps: int = 10

    # 混合精度
    fp16: bool = False
    bf16: bool = True

    # 梯度裁剪
    max_grad_norm: float = 1.0


@dataclass
class EvalConfig:
    """评测配置"""
    # 评测任务列表
    tasks: List[str] = field(default_factory=lambda: ["humaneval", "gsm8k", "mmlu"])
    # 每种任务的样本数（None表示全部）
    num_samples: Optional[int] = None
    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    # 评测模式：native（纯大模型）/ call（调用模式）/ fuse（融合模式）/ text_tool（文本级Tool Use基线）
    eval_mode: str = "call"
    # 结果保存路径
    output_path: str = "logs/eval_results.json"


@dataclass
class UniFusionConfig:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # 随机种子
    seed: int = 42
    # 是否启用调试模式（打印更多信息）
    debug: bool = True


def get_default_config() -> UniFusionConfig:
    return UniFusionConfig()
