import os
import json
import re
import subprocess
import tempfile

import torch
from qwen_asr import Qwen3ASRModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel


# ============================================================
# SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ASR_MODEL_NAME = "moorlee/qwen3-asr-0.6b-hinglish"

BASE_DIR = "/mnt/d/Final_year_project/claude_finetune"

EXTRACTOR_LORA = os.path.join(
    BASE_DIR,
    "emergency_extractor_lora"
)

SERVICE_LORA = os.path.join(
    BASE_DIR,
    "service_recommender_lora"
)

SAMPLE_RATE = 16000


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("EMERGENCY VOICE RESPONSE SYSTEM")
print("=" * 70)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"

print("=" * 70)


# ============================================================
# LOAD ASR
# ============================================================

print("\n" + "=" * 70)
print("LOADING HINGLISH ASR")
print("=" * 70)

if torch.cuda.is_available():

    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

else:

    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
    )

print("ASR loaded successfully.")


# ============================================================
# TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL FOR EXTRACTOR
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMERGENCY EXTRACTOR")
print("=" * 70)

extractor_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

extractor_base_model.config.use_cache = True

extractor_model = PeftModel.from_pretrained(
    extractor_base_model,
    EXTRACTOR_LORA,
)

extractor_model.eval()

print("Emergency extractor loaded.")


# ============================================================
# LOAD BASE MODEL FOR SERVICE RECOMMENDER
# ============================================================

print("\n" + "=" * 70)
print("LOADING SERVICE RECOMMENDER")
print("=" * 70)

service_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

service_base_model.config.use_cache = True

service_model = PeftModel.from_pretrained(
    service_base_model,
    SERVICE_LORA,
)

service_model.eval()

print("Service recommender loaded.")


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio():

    wav_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_path = wav_file.name
    wav_file.close()

    raw_path = wav_path + ".raw"

    print("\n" + "=" * 70)
    print("RECORDING STARTED")
    print("Speak Hindi / English / Hinglish")
    print("Press R + ENTER when you are DONE speaking.")
    print("=" * 70)

    process = None

    try:

        command = [
            "parec",
            "--device=RDPSource",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
        ]

        with open(raw_path, "wb") as f:

            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.DEVNULL
            )

            # Wait until user presses R + ENTER
            while True:

                stop_command = input().strip().lower()

                if stop_command == "r":
                    break

        # Stop recording
        process.terminate()

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        print("Recording stopped.")

        # ----------------------------------------------------
        # Convert RAW PCM -> WAV
        # ----------------------------------------------------

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                raw_path,
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        os.remove(raw_path)

        return wav_path

    except Exception as e:

        print("\nAUDIO RECORDING ERROR:")
        print(e)

        if process is not None:

            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        if os.path.exists(raw_path):
            os.remove(raw_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return None


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe_audio(wav_path):

    print("\nTranscribing...")

    try:

        results = asr_model.transcribe(
            audio=wav_path,
            language=None,
        )

        if results and len(results) > 0:

            text = results[0].text.strip()

            return text

        print("No transcription returned.")

    except Exception as e:

        print("\nTRANSCRIPTION ERROR:")
        print(e)

    return None


# ============================================================
# GENERATE WITH QWEN
# ============================================================

def generate(model, messages, max_new_tokens=256):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return text.strip()


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):

    if not text:
        return None

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return None

    candidate = match.group(0)

    try:

        return json.loads(candidate)

    except json.JSONDecodeError:

        try:

            start = candidate.find("{")
            end = candidate.rfind("}")

            if start >= 0 and end > start:

                return json.loads(
                    candidate[start:end + 1]
                )

        except Exception:
            pass

    return None


# ============================================================
# EMERGENCY EXTRACTION PROMPT
# ============================================================

EXTRACTOR_SYSTEM_PROMPT = """
You are an emergency information extraction agent.

Extract the following fields from the caller transcript:

emergency_type
severity
location
people_involved
injuries
additional_information

Possible emergency_type:

accident
medical_emergency
fire
domestic_emergency
missing_person
personal_safety
police_security
trapped_person
natural_disaster
unknown_other

Possible severity:

critical
high
medium
unknown

Return ONLY valid JSON.

Example:

{
  "emergency_type": "accident",
  "severity": "high",
  "location": "airport road",
  "people_involved": "one person",
  "injuries": "injured",
  "additional_information": "vehicle accident"
}

Do not invent information that is not present.
"""


# ============================================================
# EXTRACT EMERGENCY INFORMATION
# ============================================================

def extract_emergency(transcript):

    messages = [
        {
            "role": "system",
            "content": EXTRACTOR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": transcript,
        },
    ]

    raw = generate(
        extractor_model,
        messages,
        max_new_tokens=256,
    )

    result = extract_json(raw)

    if result is None:

        return {
            "emergency_type": "unknown_other",
            "severity": "unknown",
            "location": "unknown",
            "people_involved": "unknown",
            "injuries": "unknown",
            "additional_information": transcript,
        }

    return result


# ============================================================
# SERVICE RECOMMENDATION PROMPT
# ============================================================

SERVICE_SYSTEM_PROMPT = """
You are an emergency service recommendation system.

Recommend the emergency service that should be contacted based on
the caller transcript.

Possible services:

ambulance
police
fire_brigade
rescue
unknown

Recommend the intended emergency service even if the caller did not
explicitly request it.

Use the complete situation described by the caller.

Examples:

Accident with injured people -> ambulance
Breathing difficulty -> ambulance
Fire -> fire_brigade
Person being followed -> police
Person being threatened -> police
Missing person -> police
Person trapped/stuck -> rescue
Flooding requiring rescue -> rescue

Return ONLY valid JSON:

{
  "recommended_service": "...",
  "confidence": 0.0
}

Confidence must be between 0 and 1.
"""


# ============================================================
# SERVICE RECOMMENDATION
# ============================================================

def recommend_service(transcript, extracted):

    context = f"""
Caller transcript:
{transcript}

Extracted emergency information:
{json.dumps(extracted, ensure_ascii=False)}
"""

    messages = [
        {
            "role": "system",
            "content": SERVICE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": context,
        },
    ]

    raw = generate(
        service_model,
        messages,
        max_new_tokens=128,
    )

    result = extract_json(raw)

    if result is None:

        return {
            "recommended_service": "unknown",
            "confidence": 0.0,
        }

    service = result.get(
        "recommended_service",
        "unknown"
    )

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

    except Exception:

        confidence = 0.0

    valid_services = {
        "ambulance",
        "police",
        "fire_brigade",
        "rescue",
        "unknown",
    }

    if service not in valid_services:

        service = "unknown"
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return {
        "recommended_service": service,
        "confidence": confidence,
    }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_emergency(wav_path):

    # --------------------------------------------------------
    # STEP 1: ASR
    # --------------------------------------------------------

    transcript = transcribe_audio(
        wav_path
    )

    if not transcript:

        return {
            "transcription": "",
            "error": "No transcription returned."
        }

    print("\n" + "=" * 70)
    print("TRANSCRIPTION")
    print("=" * 70)

    print(transcript)

    # --------------------------------------------------------
    # STEP 2: EMERGENCY EXTRACTION
    # --------------------------------------------------------

    print("\nExtracting emergency information...")

    extracted = extract_emergency(
        transcript
    )

    # --------------------------------------------------------
    # STEP 3: SERVICE RECOMMENDATION
    # --------------------------------------------------------

    print("Recommending emergency service...")

    recommendation = recommend_service(
        transcript,
        extracted
    )

    # --------------------------------------------------------
    # STEP 4: FINAL JSON
    # --------------------------------------------------------

    final_result = {

        "transcription": transcript,

        "emergency_type":
            extracted.get(
                "emergency_type",
                "unknown_other"
            ),

        "severity":
            extracted.get(
                "severity",
                "unknown"
            ),

        "location":
            extracted.get(
                "location",
                "unknown"
            ),

        "people_involved":
            extracted.get(
                "people_involved",
                "unknown"
            ),

        "injuries":
            extracted.get(
                "injuries",
                "unknown"
            ),

        "additional_information":
            extracted.get(
                "additional_information",
                ""
            ),

        "recommended_service":
            recommendation[
                "recommended_service"
            ],

        "confidence":
            recommendation[
                "confidence"
            ],
    }

    return final_result


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 70)
print("SYSTEM READY")
print("=" * 70)

print("""
CONTROLS
----------------------------------------------------------------------
R + ENTER  -> Start recording
R + ENTER  -> Stop recording
Q + ENTER  -> Quit
----------------------------------------------------------------------
""")

while True:

    try:

        command = input("\nCommand: ").strip().lower()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if command == "q":

            print("Stopping...")
            break

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        elif command == "r":

            wav_file = record_audio()

            if wav_file is None:
                continue

            try:

                result = process_emergency(
                    wav_file
                )

                print("\n")
                print("=" * 70)
                print("FINAL EMERGENCY RESULT")
                print("=" * 70)

                print(
                    json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False
                    )
                )

                print("=" * 70)

            finally:

                if os.path.exists(wav_file):
                    os.remove(wav_file)

        else:

            print(
                "Press R + ENTER to record "
                "or Q + ENTER to quit."
            )

    except KeyboardInterrupt:

        print("\nStopping...")
        break

    except Exception as e:

        print("\nPIPELINE ERROR:")
        print(e)