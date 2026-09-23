import json
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "service_recommender_lora"
TEST_FILE = "emergency_dataset/emergency_dataset/service_recommendation/test.jsonl"

SYSTEM_PROMPT = """You are an emergency service recommendation system.

Recommend the emergency service that should be contacted based on the
caller transcript.

Possible services:
ambulance
police
fire_brigade
rescue
unknown

Recommend the intended emergency service even if the caller did not
explicitly request it.

Return ONLY valid JSON:
{"recommended_service": "...", "confidence": 0.0}
"""


def extract_json(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

print("Model ready.\n")

# Load test set
with open(TEST_FILE, encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print("Test examples:", len(test_data))

correct = 0
total = 0
results = []

for i, x in enumerate(test_data, 1):

    # Dataset format:
    # messages[1] = user transcript
    transcript = x["messages"][1]["content"]

    # Ground truth from assistant message
    expected_raw = x["messages"][2]["content"]

    try:
        expected = json.loads(expected_raw)
        expected_service = expected["required_service"]
    except:
        expected_service = "unknown"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    prediction = extract_json(generated)

    if prediction:
        predicted_service = prediction.get(
            "recommended_service",
            "unknown"
        )
        confidence = prediction.get("confidence", None)
    else:
        predicted_service = "unknown"
        confidence = None

    is_correct = predicted_service == expected_service

    if is_correct:
        correct += 1

    total += 1

    results.append({
        "transcript": transcript,
        "expected_service": expected_service,
        "predicted_service": predicted_service,
        "confidence": confidence,
        "raw_output": generated,
        "correct": is_correct
    })

    if i <= 20:
        print(f"\nTEST {i}")
        print("Transcript:", transcript)
        print("Expected :", expected_service)
        print("Predicted:", predicted_service)
        print("Confidence:", confidence)
        print("Correct  :", is_correct)

    if i % 50 == 0:
        print(f"\nProgress: {i}/{total}")

accuracy = correct / total * 100

print("\n" + "=" * 70)
print("SERVICE RECOMMENDER RESULTS")
print("=" * 70)
print(f"Correct:  {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")

with open(
    "service_recommender_test_results.jsonl",
    "w",
    encoding="utf-8"
) as f:
    for result in results:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )

print("\nSaved:")
print("service_recommender_test_results.jsonl")
