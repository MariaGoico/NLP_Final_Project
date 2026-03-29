# src/transcription/transcriber.py
import whisper
import numpy as np

# Seed global para reproducibilidad (requerido por el proyecto)
np.random.seed(42)

_model = None

def get_model(size: str = "base"):
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
        language="en",          # English
        word_timestamps=True,   # timestamps a nivel de palabra
        verbose=False,
        fp16=False,            # ← forzar FP32 siempre, evita nan en logits
    )
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": float(round(seg["start"], 2)),
            "end":   float(round(seg["end"],   2)),
            "text":  seg["text"].strip(),
        })
    return segments