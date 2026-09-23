#!/usr/bin/env python3
"""
emergency_processor.py

Standalone Emergency Information Extraction module for the AECI project.

Takes a Hindi/English code-switched (Hinglish) transcript and returns
structured JSON describing the emergency, using a small local instruction-
tuned LLM (prompted, not keyword-matched) for the understanding step.

This module is INDEPENDENT of the Qwen3 ASR pipeline. It does not touch
audio, microphones, or the existing test_hindi_asr.py. It only consumes
transcript strings and returns structured JSON.

Usage:
    python emergency_processor.py            # interactive REPL
    python emergency_processor.py --test      # run built-in test transcripts
    python emergency_processor.py --once "मैं railway station पर हूँ ..."

Environment:
    Does NOT install, upgrade, or downgrade anything. Uses whatever
    torch/transformers/numpy versions are already installed in your
    `indicconformer` conda env. Prints detected versions on startup so
    you can confirm compatibility before it tries to load a model.

Model choice:
    Defaults to Qwen/Qwen2.5-1.5B-Instruct: strong Hindi+English instruction
    following, small enough to run comfortably on a 6GB RTX 3050 in fp16
    (~3GB) or 4-bit (~1.2GB) without touching your existing ASR env.
    Override with --model <hf_repo_id> or the EMERGENCY_MODEL env var if
    you want to try Qwen2.5-3B-Instruct / Qwen3-1.7B-Instruct etc.

No hardcoded keyword rules and no hardcoded translation dictionary are
used anywhere in this file. All understanding is done by the LLM via
prompting; this file only handles I/O, prompt construction, and
JSON parsing/validation of the model's output.
"""

import argparse
import json
import os
import re
import sys
import time

# --------------------------------------------------------------------------
# Environment diagnostics (printed, never acted on automatically)
# --------------------------------------------------------------------------

def print_env_info():
    print("=" * 70)
    print("Environment check")
    print("=" * 70)
    print(f"Python:       {sys.version.split()[0]}")
    try:
        import torch
        print(f"Torch:        {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA avail:   {cuda_ok}")
        if cuda_ok:
            print(f"GPU:          {torch.cuda.get_device_name(0)}")
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"VRAM total:   {total_gb:.2f} GB")
            print(f"VRAM free:    {free_b/1e9:.2f} GB (before model load)")
    except ImportError:
        print("Torch:        NOT INSTALLED")
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")
    try:
        import numpy
        print(f"NumPy:        {numpy.__version__}")
    except ImportError:
        print("NumPy:        NOT INSTALLED")
    try:
        import bitsandbytes
        print(f"bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("bitsandbytes: not installed (4-bit quantization unavailable)")
    print("=" * 70)


# --------------------------------------------------------------------------
# Schema / prompt construction
# --------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "emergency_type",
    "severity",
    "required_service",
    "location",
    "injuries",
    "people_involved",
    "additional_information",
]

EMERGENCY_TYPES = [
    "accident", "medical_emergency", "fire", "personal_safety",
    "police_security", "missing_person", "trapped_person",
    "other", "unknown",
]
SERVICES = ["ambulance", "police", "fire_brigade", "rescue", "multiple", "unknown"]
SEVERITIES = ["low", "medium", "high", "critical", "unknown"]

SYSTEM_PROMPT = f"""You are an emergency-call information extraction assistant for an \
academic prototype. You read a single caller transcript, which may mix Hindi \
(Devanagari script) and English (Latin script) in the same sentence \
(Hinglish / code-switching), and you extract ONLY what the caller actually \
stated into a fixed JSON schema.

Rules you must follow exactly:
1. Understand the transcript as natural Hindi/English code-switched speech. \
Do not require or expect any fixed keywords.
2. Extract information ONLY if it is actually present in the transcript. \
Never guess, infer beyond what is said, or invent details.
3. If a field is not mentioned, set it to null (for location/injuries/\
people_involved/additional_information) or "unknown" (for emergency_type, \
severity, required_service).
4. severity should reflect how urgent/serious the situation sounds FROM WHAT \
WAS SAID (e.g. explicit injury, threat, bleeding, fire spreading = higher; \
vague or minor = lower). If you cannot tell, use "unknown".
5. Output VALID JSON ONLY. No markdown fences, no explanation, no extra text \
before or after the JSON object.

The JSON object must have exactly these keys:
- "emergency_type": one of {EMERGENCY_TYPES}
- "severity": one of {SEVERITIES}
- "required_service": one of {SERVICES}
- "location": string describing the stated location, or null
- "injuries": short string describing stated injuries, or null
- "people_involved": short string describing who is affected (e.g. "caller", \
"friend", "caller and one other person"), or null
- "additional_information": any other relevant detail actually stated, or null

Examples:

Transcript: "मेरा accident हो गया है मेरे friend को चोट लगी है"
JSON: {{"emergency_type": "accident", "severity": "high", "required_service": \
"ambulance", "location": null, "injuries": "friend is injured", \
"people_involved": "friend", "additional_information": null}}

Transcript: "मैं railway station पर हूँ मुझे कोई follow कर रहा है"
JSON: {{"emergency_type": "personal_safety", "severity": "high", \
"required_service": "police", "location": "railway station", "injuries": \
null, "people_involved": "caller", "additional_information": "Caller reports \
being followed."}}

Now process the next transcript the same way. Output ONLY the JSON object.
"""


def build_messages(transcript: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Transcript: "{transcript}"\nJSON:'},
    ]


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------

class EmergencyExtractor:
    def __init__(self, model_name: str, use_4bit: bool = False, device: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("WARNING: CUDA not available — running on CPU will be slow.")

        print(f"Loading tokenizer/model: {model_name} (this can take a while on first run)")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {}
        if use_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
                print("Using 4-bit quantization (bitsandbytes).")
            except ImportError:
                print("bitsandbytes not available — falling back to fp16, no quantization.")
                use_4bit = False

        if not use_4bit:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if load_kwargs.get("device_map") is None and self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded in {time.time() - t0:.1f}s on {self.device}.")
        if self.device == "cuda":
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"VRAM free after load: {free_b/1e9:.2f} GB / {total_b/1e9:.2f} GB")

    def _generate(self, transcript: str, retry_note: str = None) -> str:
        messages = build_messages(transcript)
        if retry_note:
            messages.append({"role": "user", "content": retry_note})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def extract(self, transcript: str, max_retries: int = 2) -> dict:
        raw = self._generate(transcript)
        parsed = _try_parse_json(raw)

        attempt = 0
        while parsed is None and attempt < max_retries:
            attempt += 1
            raw = self._generate(
                transcript,
                retry_note=(
                    "Your previous output was not valid JSON. Output ONLY a single "
                    "valid JSON object with exactly the required keys, nothing else."
                ),
            )
            parsed = _try_parse_json(raw)

        if parsed is None:
            return {
                "emergency_type": "unknown",
                "severity": "unknown",
                "required_service": "unknown",
                "location": None,
                "injuries": None,
                "people_involved": None,
                "additional_information": None,
                "_parse_error": True,
                "_raw_model_output": raw,
            }

        return _normalize(parsed)


# --------------------------------------------------------------------------
# JSON parsing / validation helpers
# --------------------------------------------------------------------------

def _try_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalize(parsed: dict) -> dict:
    result = {}
    for field in REQUIRED_FIELDS:
        value = parsed.get(field, None)
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ("null", "none", ""):
                value = None
        result[field] = value

    # Light normalization of the three enum-ish fields (lowercase/underscore)
    # only when the model's value already resembles the schema — we never
    # overwrite semantic content, only tidy formatting.
    for field in ("emergency_type", "severity", "required_service"):
        if isinstance(result[field], str):
            result[field] = result[field].strip().lower().replace(" ", "_")
        elif result[field] is None:
            result[field] = "unknown"

    return result


# --------------------------------------------------------------------------
# Test transcripts (data only — not logic; used to sanity-check the model)
# --------------------------------------------------------------------------

TEST_TRANSCRIPTS = [
    "मेरा accident हो गया है please ambulance bhejo",
    "मुझे police ki help chahiye koi mujhe follow kar raha hai",
    "मेरे friend ka accident hua hai hum airport road ke paas hain uske head se bleeding ho rahi hai",
    "मैं railway station पर हूँ मुझे कोई follow कर रहा है",
    "मेरा accident हो गया है मेरे friend को चोट लगी है",
    "ghar mein aag lag gayi hai jaldi fire brigade bhejo",
    "mera bhai ghar se nikla tha subah se wapas nahi aaya",
    "hum ek building ke basement mein phase gaye hain paani bhar raha hai",
    "bas thoda sa sar dard ho raha hai kuch bada nahi",
]


def run_tests(extractor: EmergencyExtractor):
    for i, transcript in enumerate(TEST_TRANSCRIPTS, 1):
        print(f"\n--- Test {i} ---")
        print(f"Transcript: {transcript}")
        t0 = time.time()
        result = extractor.extract(transcript)
        dt = time.time() - t0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"({dt:.1f}s)")


def repl(extractor: EmergencyExtractor):
    print("\nEnter a Hinglish transcript (or 'quit' to exit):")
    while True:
        try:
            transcript = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if transcript.lower() in ("quit", "exit", "q"):
            break
        if not transcript:
            continue
        t0 = time.time()
        result = extractor.extract(transcript)
        dt = time.time() - t0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Hinglish emergency information extractor")
    parser.add_argument(
        "--model",
        default=os.environ.get("EMERGENCY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="HF model repo id (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                         help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--test", action="store_true", help="Run built-in test transcripts")
    parser.add_argument("--once", type=str, default=None,
                         help="Process a single transcript and exit")
    parser.add_argument("--skip-env-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_env_check:
        print_env_info()

    extractor = EmergencyExtractor(args.model, use_4bit=args.use_4bit)

    if args.once:
        result = extractor.extract(args.once)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.test:
        run_tests(extractor)
    else:
        repl(extractor)


if __name__ == "__main__":
    main()