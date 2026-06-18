from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

MODEL_CONFIG = {
    "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "max_seq_length": 1024,
    "dtype": None,
    "load_in_4bit": True,
}

LORA_CONFIG = {
    "r": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 42,
}

TRAINING_CONFIG = {
    "output_dir": "outputs",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 5e-5,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "warmup_steps": 5,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "max_grad_norm": 0.3,
    "report_to": "none",
}

MODEL_GENERATE = {'max_new_tokens':128}