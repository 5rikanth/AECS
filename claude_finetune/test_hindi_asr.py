import os
import subprocess
import tempfile
import time

import torch
from qwen_asr import Qwen3ASRModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "moorlee/qwen3-asr-0.6b-hinglish"

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING HINGLISH ASR MODEL")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print("\nLoading model...")

if torch.cuda.is_available():
    model = Qwen3ASRModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
else:
    model = Qwen3ASRModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
    )

print("\nModel loaded successfully.")


# ============================================================
# RECORD AUDIO USING PAREC
# ============================================================

def record_audio():

    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_path = temp_wav.name
    temp_wav.close()

    print("\n" + "=" * 60)
    print(f"Recording for {RECORD_SECONDS} seconds...")
    print("Speak Hindi / Hinglish now!")
    print("=" * 60)

    try:

        # parec records raw PCM from WSLg microphone
        raw_path = wav_path + ".raw"

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

            time.sleep(RECORD_SECONDS)

            process.terminate()

            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

        # Convert raw PCM -> WAV using ffmpeg
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
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        os.remove(raw_path)

        print("Recording... done!")

        return wav_path

    except Exception as e:

        print("\nAUDIO RECORDING ERROR:")
        print(e)

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

        # IMPORTANT:
        # language=None is intentional.
        # Srota was trained with language-agnostic decoding
        # for Hinglish code-switching.

        results = model.transcribe(
            audio=wav_path,
            language=None
        )

        if results and len(results) > 0:

            text = results[0].text.strip()

            print("\n" + "=" * 60)
            print("TRANSCRIPTION")
            print("=" * 60)

            print(text)

            print("=" * 60)

            return text

        print("No transcription returned.")

    except Exception as e:

        print("\nTRANSCRIPTION ERROR:")
        print(e)

    return None


# ============================================================
# MAIN LOOP
# ============================================================

print("\n")
print("=" * 60)
print("LIVE HINDI / HINGLISH SPEECH RECOGNITION")
print("=" * 60)

print("\nCONTROLS")
print("=" * 60)
print("Press R + ENTER to record.")
print("Press Q + ENTER to quit.")
print("=" * 60)


while True:

    try:

        command = input("\nCommand: ").strip().lower()

        if command == "q":

            print("Stopping...")
            break

        elif command == "r":

            wav_file = record_audio()

            if wav_file is not None:

                transcribe_audio(wav_file)

                # Delete temporary recording
                if os.path.exists(wav_file):
                    os.remove(wav_file)

        else:

            print("Press R to record or Q to quit.")

    except KeyboardInterrupt:

        print("\nStopping...")
        break