import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "emergency_dataset/emergency_dataset/instruction_data/train.jsonl"
VAL_FILE = "emergency_dataset/emergency_dataset/instruction_data/validation.jsonl"

OUTPUT_DIR = "emergency_extractor_lora"

print("=" * 70)
print("Emergency Information Extraction - LoRA Fine-tuning")
print("=" * 70)

print("Loading datasets...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

print(dataset)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

print("Model loaded.")

# ---------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    fp16=True,

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,


    lr_scheduler_type="cosine",

    weight_decay=0.01,

    report_to="none",

    gradient_checkpointing=True,

    optim="adamw_torch",

    remove_unused_columns=False,
)

# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=peft_config,

)

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()

print("\nSaving final adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete.")
print("Adapter saved to:", OUTPUT_DIR)
