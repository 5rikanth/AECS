import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "emergency_extractor_lora"
TEST_FILE = "emergency_dataset/emergency_dataset/splits/test.jsonl"

FIELDS = [
    "emergency_type",
    "severity",
    "required_service",
    "location",
    "injuries",
    "people_involved",
    "additional_information",
]

SYSTEM_PROMPT = """You are an emergency-call information extraction system.

Extract only information explicitly stated or clearly expressed in the caller's transcript.

Return ONLY valid JSON with exactly these fields:
emergency_type
severity
required_service
location
injuries
people_involved
additional_information

Use null when information is not available.
Do not invent or infer missing information."""

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

print("Model ready.\n")

with open(TEST_FILE, encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print(f"Test examples: {len(test_data)}")

results = {field: {"correct": 0, "total": 0} for field in FIELDS}
exact_matches = 0
predictions = []

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

for item in tqdm(test_data, desc="Evaluating"):

    transcript = item["transcript"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    prediction = extract_json(text)

    if prediction is None:
        prediction = {field: None for field in FIELDS}

    # Normalize missing fields
    for field in FIELDS:
        if field not in prediction:
            prediction[field] = None

    predictions.append({
        "transcript": transcript,
        "expected": item,
        "predicted": prediction,
    })

    # Field-level exact accuracy
    all_correct = True

    for field in FIELDS:
        expected = item.get(field)
        predicted = prediction.get(field)

        results[field]["total"] += 1

        if expected == predicted:
            results[field]["correct"] += 1
        else:
            all_correct = False

    if all_correct:
        exact_matches += 1


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

for field in FIELDS:
    correct = results[field]["correct"]
    total = results[field]["total"]
    accuracy = 100 * correct / total

    print(f"{field:25s}: {accuracy:6.2f}%  ({correct}/{total})")

overall = sum(x["correct"] for x in results.values())
overall_total = sum(x["total"] for x in results.values())

print("-" * 70)
print(f"Overall field accuracy : {100 * overall / overall_total:.2f}%")
print(f"Exact 7-field matches  : {100 * exact_matches / len(test_data):.2f}%")
print(f"Exact matches          : {exact_matches}/{len(test_data)}")

# Save detailed predictions
with open("test_predictions.jsonl", "w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nSaved detailed results to:")
print("test_predictions.jsonl")
