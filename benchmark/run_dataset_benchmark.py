import os
import random
import csv
import tempfile
import soundfile as sf
import numpy as np
import nemo.collections.asr as nemo_asr
from jiwer import wer, cer

BASE = "/mnt/d/Final_year_project"
MODEL_PATH = os.path.join(BASE, "indicconformer_stt_multi_hybrid_rnnt_600m.nemo")

N_SAMPLES = 20
SEED = 42

random.seed(SEED)

# ============================================================
# Load model
# ============================================================

print("Loading IndicConformer...")

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    MODEL_PATH,
    map_location="cuda"
)

model.freeze()
model.eval()

print("Model loaded!\n")


# ============================================================
# Audio preparation
# ============================================================

def prepare_audio(audio_path, start=None, end=None):

    audio, sr = sf.read(audio_path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Extract segment
    if start is not None and end is not None:
        s = int(start * sr)
        e = int(end * sr)
        audio = audio[s:e]

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp.close()

    sf.write(temp.name, audio, sr)

    return temp.name


# ============================================================
# Transcription
# ============================================================

def transcribe(audio_path, start=None, end=None):

    temp_file = prepare_audio(
        audio_path,
        start,
        end
    )

    try:

        output = model.transcribe(
            [temp_file],
            batch_size=1,
            language_id="hi"
        )

        return str(output[0])

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)


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

    wav_scp_file = os.path.join(
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

                utt, text = parts
                transcripts[utt] = text

    wav_paths = {}

    with open(wav_scp_file, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:

                recording, filename = parts

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
# Normalize text
# ============================================================

def normalize(text):

    text = text.lower()

    # Remove punctuation
    punctuation = ".,!?;:'\"()[]{}-_/\\"

    for p in punctuation:
        text = text.replace(p, " ")

    return " ".join(text.split())


# ============================================================
# Transliteration-aware normalization
# ============================================================

def hinglish_normalize(text):

    text = normalize(text)

    replacements = {
        "गेट": "get",
        "मेल": "mail",
        "ऑल": "all",
        "न्यू": "new",
        "मैसेजेस": "messages",
        "फाइल": "file",
        "फाइल्स": "files",
        "फोल्डर": "folder",
        "सेव": "save",
        "ओपन": "open",
        "क्लिक": "click",
        "स्क्रीन": "screen",
        "विंडो": "window",
        "डॉक्यूमेंट": "document",
        "प्रेजेंटेशन": "presentation",
        "इंसर्ट": "insert",
        "फॉर्मेट": "format",
        "फॉन्ट": "font",
        "इंटरनेट": "internet",
        "कंप्यूटर": "computer",
        "सॉफ्टवेयर": "software",
        "लिबरऑफिस": "libreoffice",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# Run benchmark
# ============================================================

results = []

os.makedirs(
    os.path.join(BASE, "benchmark", "results"),
    exist_ok=True
)


# ============================================================
# Gram Vaani
# ============================================================

print("=" * 70)
print("GRAM VAANI - HINDI")
print("=" * 70)

gv = load_gramvaani()

print("Available samples:", len(gv))

gv_samples = random.sample(
    gv,
    min(N_SAMPLES, len(gv))
)

for i, (utt, audio, reference) in enumerate(gv_samples, 1):

    print(f"\n[{i}/{len(gv_samples)}] {utt}")

    try:

        prediction = transcribe(audio)

        ref = normalize(reference)
        pred = normalize(prediction)

        w = wer(ref, pred)
        c = cer(ref, pred)

        print("Reference:", reference)
        print("Prediction:", prediction)
        print(f"WER: {w:.4f} | CER: {c:.4f}")

        results.append([
            "GramVaani",
            utt,
            reference,
            prediction,
            w,
            c,
            w,
            c
        ])

    except Exception as e:

        print("ERROR:", e)


# ============================================================
# MUCS
# ============================================================

print("\n")
print("=" * 70)
print("MUCS - HINDI ENGLISH CODE-SWITCHED")
print("=" * 70)

mucs = load_mucs()

print("Available segments:", len(mucs))

mucs_samples = random.sample(
    mucs,
    min(N_SAMPLES, len(mucs))
)

for i, (utt, audio, start, end, reference) in enumerate(
    mucs_samples,
    1
):

    print(f"\n[{i}/{len(mucs_samples)}] {utt}")

    try:

        prediction = transcribe(
            audio,
            start,
            end
        )

        ref = normalize(reference)
        pred = normalize(prediction)

        strict_w = wer(ref, pred)
        strict_c = cer(ref, pred)

        ref_h = hinglish_normalize(reference)
        pred_h = hinglish_normalize(prediction)

        aware_w = wer(ref_h, pred_h)
        aware_c = cer(ref_h, pred_h)

        print("Reference:", reference)
        print("Prediction:", prediction)

        print(
            f"Strict WER: {strict_w:.4f} | "
            f"Strict CER: {strict_c:.4f}"
        )

        print(
            f"Normalized WER: {aware_w:.4f} | "
            f"Normalized CER: {aware_c:.4f}"
        )

        results.append([
            "MUCS",
            utt,
            reference,
            prediction,
            strict_w,
            strict_c,
            aware_w,
            aware_c
        ])

    except Exception as e:

        print("ERROR:", e)


# ============================================================
# Save results
# ============================================================

output = os.path.join(
    BASE,
    "benchmark",
    "results",
    "dataset_results.csv"
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
        "strict_WER",
        "strict_CER",
        "normalized_WER",
        "normalized_CER"
    ])

    writer.writerows(results)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

for dataset in ["GramVaani", "MUCS"]:

    rows = [
        r for r in results
        if r[0] == dataset
    ]

    if not rows:
        continue

    strict_wer = np.mean([r[4] for r in rows])
    strict_cer = np.mean([r[5] for r in rows])

    normalized_wer = np.mean([r[6] for r in rows])
    normalized_cer = np.mean([r[7] for r in rows])

    print(f"\n{dataset}")
    print(f"Samples: {len(rows)}")
    print(f"Strict WER:      {strict_wer:.4f}")
    print(f"Strict CER:      {strict_cer:.4f}")
    print(f"Normalized WER:  {normalized_wer:.4f}")
    print(f"Normalized CER:  {normalized_cer:.4f}")

print("\nResults saved to:")
print(output)
