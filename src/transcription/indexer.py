# src/transcription/indexer.py
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION = "segments"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

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

def index_segments(segments: list[dict], video_id: int):
    """
    segments: lista de dicts con keys:
      id, start, end, text, scene_desc
    """
    model = get_model()
    ensure_collection()
    client = get_client()

    points = []
    for seg in segments:
        # Combinar texto hablado + descripción visual
        combined = seg["text"]
        if seg.get("scene_desc"):
            combined += ". " + seg["scene_desc"]

        vector = model.encode(combined).tolist()

        points.append(PointStruct(
            id=seg["id"],
            vector=vector,
            payload={
                "video_id": video_id,
                "segment_id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "scene_desc": seg.get("scene_desc"),
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} segments for video {video_id}")