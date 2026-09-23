import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "service_recommender_lora"

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

print("Model ready.")

SYSTEM_PROMPT = """You are an emergency service recommendation system.

Recommend the emergency service that should be contacted based on the
caller transcript.

Possible services:
- ambulance
- police
- fire_brigade
- rescue
- unknown

Use the emergency situation and information in the transcript.
If the situation clearly indicates an emergency service, recommend it
even when the caller did not explicitly request that service.

Return ONLY valid JSON with exactly these fields:
recommended_service
confidence

confidence must be a number between 0 and 1.
Do not add explanations."""

TESTS = [
    "accident hua hai airport road",
    "accident hua hai airport road please ambulance bhejo",
    "ghar mein aag lag gayi hai",
    "gas cylinder mein fire lag gayi hai",
    "koi mujhe follow kar raha hai",
    "mere husband mujhe threaten kar rahe hain",
    "mera bhai missing hai subah se",
    "building mein phans gaya hoon",
    "lift mein phans gaya hoon darwaza nahi khul raha",
    "road paani mein doob gayi hai",
    "mere papa ko saans lene mein problem hai",
    "bas thoda sa headache hai kuch bada nahi",
]

def extract_json(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


print("\n" + "=" * 70)
print("SERVICE RECOMMENDATION TEST")
print("=" * 70)

for i, transcript in enumerate(TESTS, 1):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": transcript,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    result = extract_json(response)

    print(f"\nTEST {i}")
    print(f"Transcript: {transcript}")

    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("RAW OUTPUT:")
        print(response)

print("\n" + "=" * 70)
