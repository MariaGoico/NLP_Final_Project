# src/transcription/indexer.py
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION = "segments"
VECTOR_SIZE = 384

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Sentence-transformers model loaded.")
    return _model

def get_client():
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def build_text(seg: dict) -> str:
    """
    Combina texto hablado y descripción visual dando más peso al texto.
    Repetimos el texto 2 veces para que tenga más peso semántico.
    """
    text = seg["text"].strip()
    scene = seg.get("scene_desc", "") or ""
    if scene:
        return f"{text} {text} {scene}"
    return text

def index_segments(segments: list[dict], video_id: int):
    model = get_model()
    ensure_collection()
    client = get_client()

    texts = [build_text(seg) for seg in segments]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False)

    points = []
    for seg, vector in zip(segments, vectors):
        points.append(PointStruct(
            id=seg["id"],
            vector=vector.tolist(),
            payload={
                "video_id": video_id,
                "segment_id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "scene_desc": seg.get("scene_desc"),
                "combined_text": build_text(seg),
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} segments for video {video_id}")