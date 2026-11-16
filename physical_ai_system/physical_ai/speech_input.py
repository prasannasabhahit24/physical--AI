"""Optional voice input. Typed commands work without this module.

Needs:  pip install SpeechRecognition pyaudio
Uses the free Google Web Speech endpoint by default (needs internet). For an offline
alternative replace `recognize_google` with `recognize_whisper` (pip install openai-whisper).
"""


def listen_once(language: str = "en-IN", timeout: float = 6.0, phrase_limit: float = 8.0) -> str:
    import speech_recognition as sr
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = rec.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
    return rec.recognize_google(audio, language=language)
