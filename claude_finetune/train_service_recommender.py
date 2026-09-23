import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "emergency_dataset/emergency_dataset/service_recommendation/train.jsonl"
VAL_FILE = "emergency_dataset/emergency_dataset/service_recommendation/validation.jsonl"

OUTPUT_DIR = "service_recommender_lora"

print("=" * 70)
print("Emergency Service Recommendation - LoRA Fine-tuning")
print("=" * 70)

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

print("Loading datasets...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

print(dataset)

# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

print("Model loaded.")

# ------------------------------------------------------------
# FORMAT DATA
# ------------------------------------------------------------

def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ------------------------------------------------------------
# LORA
# ------------------------------------------------------------

lora_config = LoraConfig(
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

# ------------------------------------------------------------
# TRAINING CONFIG
# ------------------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=5e-5,

    warmup_steps=50,

    weight_decay=0.01,

    logging_steps=10,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    fp16=True,

    gradient_checkpointing=True,

    report_to="none",

    max_length=512,

    dataset_text_field="text",

    packing=False,
)

# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------

print("\nCreating trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=lora_config,

    formatting_func=format_example,
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING SERVICE RECOMMENDATION TRAINING")
print("=" * 70)

trainer.train()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print("\nSaving adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Adapter saved to: {OUTPUT_DIR}")
