import speech_recognition as sr
import nemo.collections.asr as nemo_asr

MODEL = "/mnt/d/Final_year_project/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

# Load IndicConformer once
print("Loading IndicConformer...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    MODEL, map_location="cuda"
)
model.freeze()
model.eval()
print("Model loaded!\n")


def google_asr(file):
    r = sr.Recognizer()

    with sr.AudioFile(file) as source:
        audio = r.record(source)

    try:
        return r.recognize_google(audio, language="en-IN")
    except Exception as e:
        return f"ERROR: {e}"


def indic_asr(file):
    try:
        result = model.transcribe(
            [file],
            batch_size=1,
            language_id="hi"
        )
        return str(result[0])
    except Exception as e:
        return f"ERROR: {e}"


files = {
    "ENGLISH": "/mnt/d/Final_year_project/audio/English.wav",
    "HINDI": "/mnt/d/Final_year_project/audio/Hindi.wav",
    "HINGLISH": "/mnt/d/Final_year_project/audio/Hinglish.wav"
}

for name, file in files.items():

    print("=" * 60)
    print(name)
    print("=" * 60)

    if name == "ENGLISH":
        text = google_asr(file)
    else:
        text = indic_asr(file)

    print("Transcript:")
    print(text)
    print()
