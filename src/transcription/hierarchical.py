# src/transcription/hierarchical.py
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION_WINDOWS = "windows"
COLLECTION_VIDEOS = "videos_summary"
VECTOR_SIZE = 384
WINDOW_SIZE = 4  # segmentos por ventana


def get_client():
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )


def ensure_collections():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    for name in [COLLECTION_WINDOWS, COLLECTION_VIDEOS]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )


def build_windows(segments: list[dict]) -> list[dict]:
    """
    Agrupa segmentos consecutivos en ventanas de WINDOW_SIZE.
    Cada ventana tiene: id, segment_ids, start, end, text combinado.
    """
    windows = []
    for i in range(0, len(segments), WINDOW_SIZE):
        chunk = segments[i:i + WINDOW_SIZE]
        window = {
            "id": i // WINDOW_SIZE,
            "segment_ids": [s["id"] for s in chunk],
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(s["text"] for s in chunk),
            "scene_descs": " ".join(s["scene_desc"] or "" for s in chunk),
        }
        windows.append(window)
    return windows


def index_hierarchical(segments: list[dict], video_id: int, model: SentenceTransformer):
    ensure_collections()
    client = get_client()

    # ── Nivel 2: ventanas temporales ─────────────────────────────────────────
    windows = build_windows(segments)
    window_texts = [
        f"{w['text']} {w['scene_descs']}".strip()
        for w in windows
    ]
    window_vectors = model.encode(window_texts, batch_size=32, show_progress_bar=False)

    window_points = []
    for w, vec in zip(windows, window_vectors):
        # Usamos un ID único global combinando video_id y window id
        point_id = int(f"{video_id}{w['id']:04d}")
        window_points.append(PointStruct(
            id=point_id,
            vector=vec.tolist(),
            payload={
                "video_id": video_id,
                "window_id": w["id"],
                "segment_ids": w["segment_ids"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
                "scene_descs": w["scene_descs"],
            }
        ))
    client.upsert(collection_name=COLLECTION_WINDOWS, points=window_points)
    print(f"Indexed {len(window_points)} windows for video {video_id}")

    # ── Nivel 3: resumen completo del vídeo ──────────────────────────────────
    full_text = " ".join(s["text"] for s in segments)
    full_scenes = " ".join(s["scene_desc"] or "" for s in segments)
    video_text = f"{full_text} {full_scenes}".strip()

    video_vector = model.encode(video_text).tolist()
    client.upsert(
        collection_name=COLLECTION_VIDEOS,
        points=[PointStruct(
            id=video_id,
            vector=video_vector,
            payload={
                "video_id": video_id,
                "full_text": full_text,
                "full_scenes": full_scenes,
                "num_segments": len(segments),
                "start": segments[0]["start"],
                "end": segments[-1]["end"],
            }
        )]
    )
    print(f"Indexed video summary for video {video_id}")