# src/transcription/main.py
import os, tempfile, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .db import init_db, get_conn
from .storage import upload_file
from .transcriber import transcribe

app = FastAPI(title="Transcription Service")

@app.on_event("startup")
def startup():
    init_db()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. Guardar temporalmente en disco
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. Subir a MinIO
        object_name = f"{uuid.uuid4()}{suffix}"
        minio_path = upload_file(tmp_path, object_name)

        # 3. Registrar vídeo en PostgreSQL
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO videos (filename, minio_path, status) VALUES (%s, %s, 'transcribing') RETURNING id",
                    (file.filename, minio_path),
                )
                video_id = cur.fetchone()[0]
            conn.commit()

        # 4. Transcribir con Whisper (síncrono por ahora)
        segments = transcribe(tmp_path)

        # 5. Guardar segmentos en PostgreSQL
        with get_conn() as conn:
            with conn.cursor() as cur:
                for seg in segments:
                    cur.execute(
                        """INSERT INTO segments (video_id, start_s, end_s, text)
                           VALUES (%s, %s, %s, %s)""",
                        (video_id, seg["start"], seg["end"], seg["text"]),
                    )
                cur.execute(
                    "UPDATE videos SET status='transcribed' WHERE id=%s",
                    (video_id,),
                )
            conn.commit()

        return JSONResponse({
            "video_id": video_id,
            "segments": len(segments),
            "preview": segments[:3],   # primeros 3 para verificar
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
                "SELECT id, start_s, end_s, text FROM segments WHERE video_id=%s ORDER BY start_s",
                (video_id,),
            )
            rows = cur.fetchall()
    return [{"id": r[0], "start": r[1], "end": r[2], "text": r[3]} for r in rows]

@app.get("/health")
def health():
    return {"status": "ok"}