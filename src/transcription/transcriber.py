# src/transcription/transcriber.py
import whisper
import numpy as np

# Seed global para reproducibilidad (requerido por el proyecto)
np.random.seed(42)

_model = None

def get_model(size: str = "large-v3"):
    global _model
    if _model is None:
        _model = whisper.load_model(size)
    return _model

def transcribe(video_path: str) -> list[dict]:
    """
    Devuelve lista de segmentos:
    [{"start": 0.0, "end": 4.2, "text": "Hola, esto es..."}, ...]
    """
    model = get_model()
    result = model.transcribe(
        video_path,
        language=None,          # autodetect (español, inglés, etc.)
        word_timestamps=True,   # timestamps a nivel de palabra
        verbose=False,
    )
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        })
    return segments