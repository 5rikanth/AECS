import speech_recognition as sr
import nemo.collections.asr as nemo_asr
from jiwer import wer, cer

BASE = "/mnt/d/Final_year_project"
MODEL = f"{BASE}/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

files = {
    "English": f"{BASE}/benchmark/audio/English.wav",
    "Hindi": f"{BASE}/benchmark/audio/Hindi.wav",
    "Hinglish": f"{BASE}/benchmark/audio/Hinglish.wav"
}

# Load reference transcripts
references = {}

with open(f"{BASE}/benchmark/references.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Ignore blank/malformed lines
        if not line or "|" not in line:
            continue

        name, text = line.split("|", 1)
        references[name] = text


# -------------------------
# Google ASR
# -------------------------
def google_asr(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    return recognizer.recognize_google(
        audio,
        language="en-IN"
    )


# -------------------------
# IndicConformer
# -------------------------
print("Loading IndicConformer...")

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    MODEL,
    map_location="cuda"
)

model.freeze()
model.eval()

print("Model loaded successfully!\n")


def indic_asr(audio_file):
    result = model.transcribe(
        [audio_file],
        batch_size=1,
        language_id="hi"
    )

    return str(result[0])


# -------------------------
# Benchmark
# -------------------------
results = {}

for language, audio_file in files.items():

    print("=" * 60)
    print(language)
    print("=" * 60)

    try:

        # English → Google ASR
        if language == "English":
            prediction = google_asr(audio_file)

        # Hindi/Hinglish → IndicConformer
        else:
            prediction = indic_asr(audio_file)

        reference = references[language]

        results[language] = prediction

        print("\nREFERENCE:")
        print(reference)

        print("\nPREDICTION:")
        print(prediction)

        w = wer(reference, prediction)
        c = cer(reference, prediction)

        print("\nWER:", round(w, 4))
        print("CER:", round(c, 4))

    except Exception as e:
        print("\nERROR:", e)


# -------------------------
# Save results
# -------------------------
output = f"{BASE}/benchmark/results/baseline_results.txt"

with open(output, "w", encoding="utf-8") as f:

    for language, prediction in results.items():

        reference = references[language]

        f.write("=" * 60 + "\n")
        f.write(f"{language}\n")
        f.write("=" * 60 + "\n")

        f.write(f"Reference:\n{reference}\n\n")
        f.write(f"Prediction:\n{prediction}\n\n")
        f.write(f"WER: {wer(reference, prediction):.4f}\n")
        f.write(f"CER: {cer(reference, prediction):.4f}\n\n")


print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
print(f"Results saved to:\n{output}")
