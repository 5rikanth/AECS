import speech_recognition as sr

audio_file = "/mnt/d/Final_year_project/audio/English.wav"

r = sr.Recognizer()

with sr.AudioFile(audio_file) as source:
    audio = r.record(source)

try:
    text = r.recognize_google(audio, language="en-IN")
    print("English transcription:")
    print(text)

except sr.UnknownValueError:
    print("Could not understand the audio.")

except sr.RequestError as e:
    print("Google ASR error:", e)
