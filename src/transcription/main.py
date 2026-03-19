# src/transcription/main.py
import os, tempfile, uuid, subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .db import init_db, get_conn
from .storage import upload_file, upload_frame
from .transcriber import transcribe
from .vision import describe_frame
from .indexer import index_segments, get_client, COLLECTION, get_model as get_embedder
from .llm import answer as llm_answer

app = FastAPI(title="Transcription Service")

@app.on_event("startup")
def startup():
    init_db()

def extract_frame(video_path: str, timestamp: float, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        object_name = f"{uuid.uuid4()}{suffix}"
        minio_path = upload_file(tmp_path, object_name)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO videos (filename, minio_path, status) VALUES (%s, %s, 'transcribing') RETURNING id",
                    (file.filename, minio_path),
                )
                video_id = cur.fetchone()[0]
            conn.commit()

        segments = transcribe(tmp_path)

        saved_segments = []
        with get_conn() as conn:
            with conn.cursor() as cur:
                for seg in segments:
                    mid = round((seg["start"] + seg["end"]) / 2, 2)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_frame:
                        tmp_frame_path = tmp_frame.name

                    frame_path = None
                    scene_desc = None

                    try:
                        extract_frame(tmp_path, mid, tmp_frame_path)
                        frame_path = upload_frame(tmp_frame_path, f"{uuid.uuid4()}.jpg")
                        scene_desc = describe_frame(tmp_frame_path)
                    except Exception as e:
                        print(f"Frame/vision error at {mid}s: {e}")
                    finally:
                        if os.path.exists(tmp_frame_path):
                            os.unlink(tmp_frame_path)

                    cur.execute(
                        """INSERT INTO segments (video_id, start_s, end_s, text, frame_path, scene_desc)
                           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                        (video_id, seg["start"], seg["end"], seg["text"], frame_path, scene_desc),
                    )
                    segment_id = cur.fetchone()[0]
                    saved_segments.append({
                        "id": segment_id,
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "scene_desc": scene_desc,
                    })

                cur.execute("UPDATE videos SET status='indexed' WHERE id=%s", (video_id,))
            conn.commit()

        index_segments(saved_segments, video_id)

        return JSONResponse({
            "video_id": video_id,
            "segments": len(saved_segments),
            "preview": segments[:3],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.get("/videos/{video_id}/segments")
def get_segments(video_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, start_s, end_s, text, frame_path, scene_desc FROM segments WHERE video_id=%s ORDER BY start_s",
                (video_id,),
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "start": r[1], "end": r[2], "text": r[3], "frame_path": r[4], "scene_desc": r[5]}
        for r in rows
    ]

@app.post("/query")
def query(payload: dict):
    question = payload.get("question", "")
    video_id = payload.get("video_id")  # opcional: filtrar por vídeo
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # 1. Embedding de la pregunta
    embedder = get_embedder()
    vector = embedder.encode(question).tolist()

    # 2. Buscar en Qdrant — traemos más candidatos para luego reranquear
    client = get_client()
    query_kwargs = dict(
        collection_name=COLLECTION,
        query=vector,
        limit=8,  # traemos 8, luego filtramos a los mejores
        with_payload=True,
        score_threshold=0.2,  # descartamos resultados muy poco relevantes
    )
    if video_id:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_kwargs["query_filter"] = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )

    results = client.query_points(**query_kwargs).points

    # 3. Reranking — ordenar por score y quedarnos con top 4
    results = sorted(results, key=lambda r: r.score, reverse=True)[:4]

    # 4. Construir contexto ordenado por timestamp
    context_segments = sorted([
        {
            "start": r.payload["start"],
            "end": r.payload["end"],
            "text": r.payload["text"],
            "scene_desc": r.payload.get("scene_desc"),
            "score": round(r.score, 3),
        }
        for r in results
    ], key=lambda x: x["start"])

    # 5. Generar respuesta con LLaMA
    response = llm_answer(question, context_segments)

    return {
        "question": question,
        "answer": response,
        "sources": context_segments,
    }

@app.get("/health")
def health():
    return {"status": "ok"}