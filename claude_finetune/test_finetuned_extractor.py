import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "emergency_extractor_lora"

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
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

print("Model ready.\n")

tests = [
    "मेरा accident हो गया है please ambulance bhejo",
    "मुझे police ki help chahiye koi mujhe follow kar raha hai",
    "मेरे friend ka accident hua hai hum airport road ke paas hain uske head se bleeding ho rahi hai",
    "मैं railway station पर हूँ मुझे कोई follow कर रहा है",
    "मेरा accident हो गया है मेरे friend को चोट लगी है",
    "ghar mein aag lag gayi hai jaldi fire brigade bhejo",
    "mera bhai ghar se nikla tha subah se wapas nahi aaya",
    "hum ek building ke basement mein phase gaye hain paani bhar raha hai",
    "bas thoda sa sar dard ho raha hai kuch bada nahi"
]

for i, transcript in enumerate(tests, 1):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript}
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    print("=" * 70)
    print(f"TEST {i}")
    print("Transcript:", transcript)
    print("Output:")
    print(response.strip())
    print()

