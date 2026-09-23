import os
import json
import subprocess
import tempfile
import sys

import torch
from qwen_asr import Qwen3ASRModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emergency_processor import EmergencyExtractor, print_env_info  # noqa: E402
from service_agent import ServiceAgent                              # noqa: E402
from fusion_agent import fuse                                       # noqa: E402


# ============================================================
# SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ASR_MODEL_NAME = "moorlee/qwen3-asr-0.6b-hinglish"

# NOTE: EXTRACTOR_LORA / SERVICE_LORA are no longer used. Both the
# extractor and the service recommender now run as prompted (not
# fine-tuned) calls against ONE shared base-model instance, which is
# why there's only one model load below instead of two.

SAMPLE_RATE = 16000


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("EMERGENCY VOICE RESPONSE SYSTEM")
print("=" * 70)

print_env_info()  # torch/transformers/CUDA/VRAM diagnostic, same as emergency_processor.py


# ============================================================
# LOAD ASR  (unchanged from your original script)
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
# LOAD EXTRACTOR + SERVICE AGENT (single shared model, no LoRA)
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMERGENCY EXTRACTOR + SERVICE AGENT (shared model)")
print("=" * 70)

extractor = EmergencyExtractor(BASE_MODEL)   # one base-model load
service_agent = ServiceAgent(extractor)       # reuses extractor.model / .tokenizer — no second load

print("Extractor + service agent ready.")


# ============================================================
# RECORD AUDIO  (unchanged from your original script)
# ============================================================

def record_audio():
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
            "parec", "--device=RDPSource", "--format=s16le",
            "--rate=16000", "--channels=1",
        ]
        with open(raw_path, "wb") as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.DEVNULL)
            while True:
                stop_command = input().strip().lower()
                if stop_command == "r":
                    break

        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        print("Recording stopped.")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
             "-i", raw_path, wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
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
# TRANSCRIBE  (unchanged from your original script)
# ============================================================

def transcribe_audio(wav_path):
    print("\nTranscribing...")
    try:
        results = asr_model.transcribe(audio=wav_path, language=None)
        if results and len(results) > 0:
            return results[0].text.strip()
        print("No transcription returned.")
    except Exception as e:
        print("\nTRANSCRIPTION ERROR:")
        print(e)
    return None


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_emergency(wav_path):
    transcript = transcribe_audio(wav_path)
    if not transcript:
        return {"transcription": "", "error": "No transcription returned."}

    print("\n" + "=" * 70)
    print("TRANSCRIPTION")
    print("=" * 70)
    print(transcript)

    print("\nExtracting emergency information...")
    extraction = extractor.extract(transcript)

    print("Recommending emergency service...")
    service_rec = service_agent.recommend(transcript, extraction)

    final_result = fuse(transcript, extraction, service_rec)
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

        if command == "q":
            print("Stopping...")
            break

        elif command == "r":
            wav_file = record_audio()
            if wav_file is None:
                continue
            try:
                result = process_emergency(wav_file)
                print("\n")
                print("=" * 70)
                print("FINAL EMERGENCY RESULT")
                print("=" * 70)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("=" * 70)
            finally:
                if os.path.exists(wav_file):
                    os.remove(wav_file)

        else:
            print("Press R + ENTER to record or Q + ENTER to quit.")

    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print("\nPIPELINE ERROR:")
        print(e)
