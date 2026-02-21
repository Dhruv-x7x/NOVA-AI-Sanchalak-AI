"""
SANCHALAK AI - Fine-Tuning Configuration
==============================================
Configuration for training custom clinical trial AI models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 16  # Rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Memory optimization
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    fp16: bool = True  # Use 16-bit precision
    
    # Saving
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Evaluation
    eval_steps: int = 50
    logging_steps: int = 10


@dataclass
class FineTuningConfig:
    """Main fine-tuning configuration."""
    
    # Model
    base_model: str = "unsloth/llama-3.1-8b-bnb-4bit"  # 4-bit quantized for efficiency
    output_model_name: str = "sanchalak-ai-v1"
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "training_data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    
    # Configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Dataset settings
    train_split: float = 0.9
    validation_split: float = 0.1
    
    # Clinical trial specific
    domain_focus: List[str] = field(default_factory=lambda: [
        "data_quality",
        "issue_resolution",
        "query_management",
        "site_performance",
        "patient_safety"
    ])
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Singleton
_config: Optional[FineTuningConfig] = None


def get_finetuning_config() -> FineTuningConfig:
    """Get fine-tuning configuration."""
    global _config
    if _config is None:
        _config = FineTuningConfig()
    return _config


# System prompts for different training scenarios
TRAINING_SYSTEM_PROMPTS = {
    "data_quality": """You are an expert clinical trial data quality analyst. 
You analyze patient data quality indicators, identify issues, and provide 
actionable recommendations. Always cite specific data points and metrics.
Responses should be precise, data-driven, and regulatory-compliant.""",

    "issue_resolution": """You are a clinical trial issue resolution specialist.
You diagnose data quality issues, determine root causes, and recommend
evidence-based solutions. Prioritize patient safety and regulatory compliance.
Provide step-by-step resolution guidance.""",

    "query_management": """You are a clinical trial query management expert.
You draft professional, clear query responses that address data discrepancies.
Your responses should be suitable for regulatory submission and maintain
audit trail integrity.""",

    "site_performance": """You are a clinical trial site performance analyst.
You evaluate site metrics, identify performance trends, and recommend
interventions for underperforming sites. Focus on enrollment, data quality,
and protocol compliance.""",

    "patient_safety": """You are a clinical trial safety analyst.
You identify potential safety signals, prioritize SAE reconciliation,
and ensure timely regulatory reporting. Patient safety is your highest priority.
Never make medical decisions - only flag for medical review."""
}
