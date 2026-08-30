import os
import random
import csv
import time
import soundfile as sf
import numpy as np
import speech_recognition as sr
from jiwer import wer, cer

BASE = "/mnt/d/Final_year_project"
N_SAMPLES = 20
SEED = 42

random.seed(SEED)

recognizer = sr.Recognizer()


# ============================================================
# Audio preparation
# ============================================================

def prepare_audio(audio_path, start=None, end=None):

    audio, sr_rate = sf.read(audio_path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Extract MUCS segment
    if start is not None and end is not None:
        s = int(start * sr_rate)
        e = int(end * sr_rate)
        audio = audio[s:e]

    temp = os.path.join(BASE, "benchmark", "google_temp.wav")

    sf.write(temp, audio, sr_rate)

    return temp


# ============================================================
# Google ASR
# ============================================================

def google_transcribe(audio_path, start=None, end=None):

    temp = prepare_audio(audio_path, start, end)

    try:

        with sr.AudioFile(temp) as source:
            audio = recognizer.record(source)

        # Google Web Speech API
        text = recognizer.recognize_google(
            audio,
            language="hi-IN"
        )

        return text

    except sr.UnknownValueError:
        return ""

    except sr.RequestError as e:
        print("Google API error:", e)
        return ""

    finally:

        if os.path.exists(temp):
            os.remove(temp)


# ============================================================
# Text normalization
# ============================================================

def normalize(text):

    text = text.lower()

    punctuation = ".,!?;:'\"()[]{}-_/\\"

    for p in punctuation:
        text = text.replace(p, " ")

    return " ".join(text.split())


# ============================================================
# Load Gram Vaani
# ============================================================

def load_gramvaani():

    folder = os.path.join(
        BASE,
        "GV_Eval_3h",
        "GV_Eval_3h"
    )

    text_file = os.path.join(folder, "text")
    scp_file = os.path.join(folder, "mp3.scp")

    transcripts = {}

    with open(text_file, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                utt, text = parts
                transcripts[utt] = text

    audio_paths = {}

    with open(scp_file, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:

                utt, path = parts

                path = path.strip().lstrip("./")

                audio_paths[utt] = os.path.join(
                    folder,
                    path
                )

    data = []

    for utt in transcripts:

        if utt in audio_paths:

            if os.path.exists(audio_paths[utt]):

                data.append(
                    (
                        utt,
                        audio_paths[utt],
                        transcripts[utt]
                    )
                )

    return data


# ============================================================
# Load MUCS
# ============================================================

def load_mucs():

    folder = os.path.join(
        BASE,
        "Hindi-English_test",
        "test"
    )

    text_file = os.path.join(
        folder,
        "transcripts",
        "text"
    )

    wav_scp = os.path.join(
        folder,
        "transcripts",
        "wav.scp"
    )

    segments_file = os.path.join(
        folder,
        "transcripts",
        "segments"
    )

    transcripts = {}

    with open(text_file, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

    wav_paths = {}

    with open(wav_scp, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:

                recording = parts[0]
                filename = parts[1]

                wav_paths[recording] = os.path.join(
                    folder,
                    filename
                )

    data = []

    with open(segments_file, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 4:
                continue

            utt = parts[0]
            recording = parts[1]
            start = float(parts[2])
            end = float(parts[3])

            if (
                utt in transcripts
                and recording in wav_paths
                and os.path.exists(wav_paths[recording])
            ):

                data.append(
                    (
                        utt,
                        wav_paths[recording],
                        start,
                        end,
                        transcripts[utt]
                    )
                )

    return data


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Gram Vaani
# ============================================================

print("=" * 70)
print("GOOGLE ASR - GRAM VAANI HINDI")
print("=" * 70)

gv = load_gramvaani()

samples = random.sample(
    gv,
    min(N_SAMPLES, len(gv))
)

for i, (utt, audio, reference) in enumerate(samples, 1):

    print(f"\n[{i}/{len(samples)}] {utt}")

    prediction = google_transcribe(audio)

    ref = normalize(reference)
    pred = normalize(prediction)

    w = wer(ref, pred)
    c = cer(ref, pred)

    print("Reference :", reference)
    print("Prediction:", prediction)
    print(f"WER: {w:.4f} | CER: {c:.4f}")

    results.append([
        "GramVaani",
        utt,
        reference,
        prediction,
        w,
        c
    ])

    time.sleep(1)


# ============================================================
# MUCS
# ============================================================

print("\n")
print("=" * 70)
print("GOOGLE ASR - MUCS HINDI ENGLISH")
print("=" * 70)

mucs = load_mucs()

samples = random.sample(
    mucs,
    min(N_SAMPLES, len(mucs))
)

for i, (utt, audio, start, end, reference) in enumerate(
    samples,
    1
):

    print(f"\n[{i}/{len(samples)}] {utt}")

    prediction = google_transcribe(
        audio,
        start,
        end
    )

    ref = normalize(reference)
    pred = normalize(prediction)

    w = wer(ref, pred)
    c = cer(ref, pred)

    print("Reference :", reference)
    print("Prediction:", prediction)
    print(f"WER: {w:.4f} | CER: {c:.4f}")

    results.append([
        "MUCS",
        utt,
        reference,
        prediction,
        w,
        c
    ])

    time.sleep(1)


# ============================================================
# Save
# ============================================================

output = os.path.join(
    BASE,
    "benchmark",
    "results",
    "google_results.csv"
)

with open(
    output,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "dataset",
        "utterance_id",
        "reference",
        "prediction",
        "WER",
        "CER"
    ])

    writer.writerows(results)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("GOOGLE ASR BENCHMARK COMPLETE")
print("=" * 70)

for dataset in ["GramVaani", "MUCS"]:

    rows = [
        r for r in results
        if r[0] == dataset
    ]

    if rows:

        avg_wer = np.mean([r[4] for r in rows])
        avg_cer = np.mean([r[5] for r in rows])

        print(
            f"{dataset}: "
            f"{len(rows)} samples | "
            f"WER: {avg_wer:.4f} | "
            f"CER: {avg_cer:.4f}"
        )

print("\nSaved to:")
print(output)
