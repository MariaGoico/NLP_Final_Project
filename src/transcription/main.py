# src/transcription/main.py
import os, tempfile, uuid, subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .db import init_db, get_conn
from .storage import upload_file, upload_frame
from .transcriber import transcribe
from .vision import describe_frame
from .indexer import index_segments
from .llm import answer as llm_answer
from .indexer import get_client, COLLECTION
from sentence_transformers import SentenceTransformer

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
        # 1. Subir vídeo a MinIO
        object_name = f"{uuid.uuid4()}{suffix}"
        minio_path = upload_file(tmp_path, object_name)

        # 2. Registrar vídeo en PostgreSQL
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO videos (filename, minio_path, status) VALUES (%s, %s, 'transcribing') RETURNING id",
                    (file.filename, minio_path),
                )
                video_id = cur.fetchone()[0]
            conn.commit()

        # 3. Transcribir con Whisper
        segments = transcribe(tmp_path)

        # 4. Extraer frames, describir y guardar segmentos
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

                cur.execute(
                    "UPDATE videos SET status='indexed' WHERE id=%s",
                    (video_id,),
                )
            conn.commit()

        # 5. Indexar en Qdrant
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
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # 1. Embedding de la pregunta
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode(question).tolist()

    # 2. Buscar en Qdrant
    client = get_client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=4,
    )

    # 3. Construir contexto
    context_segments = []
    for r in results:
        context_segments.append({
            "start": r.payload["start"],
            "end": r.payload["end"],
            "text": r.payload["text"],
            "scene_desc": r.payload.get("scene_desc"),
        })

    # 4. Generar respuesta con LLaMA
    response = llm_answer(question, context_segments)

    return {
        "question": question,
        "answer": response,
        "sources": context_segments,
    }

@app.get("/health")
def health():
    return {"status": "ok"}