# src/transcription/main.py
import os, tempfile, uuid, subprocess
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .db import init_db, get_conn
from .storage import upload_file, upload_frame
from .transcriber import transcribe
from .vision import describe_frame
from .indexer import index_segments, get_client, COLLECTION, get_model as get_embedder
from .hierarchical import index_hierarchical, COLLECTION_WINDOWS, COLLECTION_VIDEOS
from .llm import answer as llm_answer

app = FastAPI(title="Transcription Service")


@app.on_event("startup")
def startup():
    init_db()
    from .indexer import get_model as get_embedder
    from .llm import get_model as get_llm
    print("Precargando modelos...")
    get_embedder()
    get_llm()
    print("Modelos precargados — servicio listo.")


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

        # Nivel 1 — segmentos, reutilizamos el modelo devuelto
        model = index_segments(saved_segments, video_id)

        # Niveles 2 y 3 — ventanas y resumen completo
        index_hierarchical(saved_segments, video_id, model)

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
    video_id = payload.get("video_id")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    embedder = get_embedder()
    vector = embedder.encode(question).tolist()
    client = get_client()

    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, HasIdCondition

    # ── Nivel 3: resumen completo siempre ────────────────────────────────────
    video_filter = None
    if video_id:
        video_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )

    video_hits = client.query_points(
        collection_name=COLLECTION_VIDEOS,
        query=vector,
        limit=3,
        query_filter=video_filter,
        with_payload=True,
    ).points

    # Contexto global siempre presente
    global_context = None
    if video_hits:
        global_context = {
            "start": video_hits[0].payload["start"],
            "end": video_hits[0].payload["end"],
            "text": video_hits[0].payload["full_text"],
            "scene_desc": video_hits[0].payload.get("full_scenes"),
            "score": round(video_hits[0].score, 3),
            "level": "video_summary",
        }

    # ── Nivel 2: ventanas ─────────────────────────────────────────────────────
    relevant_video_ids = [h.payload["video_id"] for h in video_hits]
    if video_id and video_id not in relevant_video_ids:
        relevant_video_ids.append(video_id)

    window_filter = Filter(
        must=[FieldCondition(key="video_id", match=MatchAny(any=relevant_video_ids))]
    )
    window_hits = client.query_points(
        collection_name=COLLECTION_WINDOWS,
        query=vector,
        limit=4,
        query_filter=window_filter,
        with_payload=True,
        score_threshold=0.15,
    ).points

    # ── Nivel 1: segmentos ────────────────────────────────────────────────────
    candidate_segment_ids = []
    for w in window_hits:
        candidate_segment_ids.extend(w.payload["segment_ids"])
    candidate_segment_ids = list(set(candidate_segment_ids))

    segment_hits = []
    if candidate_segment_ids:
        segment_filter = Filter(
            must=[HasIdCondition(has_id=candidate_segment_ids)]
        )
        segment_hits = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=4,
            query_filter=segment_filter,
            with_payload=True,
            score_threshold=0.20,
        ).points
        segment_hits = sorted(segment_hits, key=lambda r: r.score, reverse=True)[:4]

    specific_segments = sorted([
        {
            "start": r.payload["start"],
            "end": r.payload["end"],
            "text": r.payload["text"],
            "scene_desc": r.payload.get("scene_desc"),
            "score": round(r.score, 3),
            "level": "segment",
        }
        for r in segment_hits
    ], key=lambda x: x["start"])

    # ── Construir contexto final: resumen global + segmentos específicos ──────
    context_segments = []
    if global_context:
        context_segments.append(global_context)
    context_segments.extend(specific_segments)

    response = llm_answer(question, context_segments)

    # Determinar nivel real usado
    if not segment_hits and not window_hits and global_context:
        level_used = "video_summary_shortcut"  # solo usó nivel 3
    elif not segment_hits and window_hits:
        level_used = "hierarchical"            # llegó hasta nivel 2
    else:
        level_used = "hierarchical_full"       # llegó hasta nivel 1


    return {
        "question": question,
        "answer": response,
        "sources": specific_segments,  # en UI solo mostramos segmentos específicos
        "retrieval_levels_used": {
            "videos": len(video_hits),
            "windows": len(window_hits),
            "segments": len(segment_hits),
            "level_used": level_used,
        }
    }

@app.get("/videos")
# def list_videos():
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute("SELECT id, filename, minio_path, status, created_at FROM videos ORDER BY created_at DESC")
#             rows = cur.fetchall()
#     return [
#         {
#             "id": r[0], 
#             "filename": r[1], 
#             "minio_path": r[2], 
#             "status": r[3], 
#             "created_at": str(r[4])
#         } for r in rows
#     ]
def list_videos():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Seleccionamos filename (que es donde guardas el title) y el id
            cur.execute("SELECT id, filename, youtube_url, youtube_id, description, release_year, genre FROM videos WHERE status = 'indexed'")
            rows = cur.fetchall()
            
    return [
        {
            "id": r[0], 
            "title": r[1], 
            "youtube_url": r[2], 
            "youtube_id": r[3],
            "description": r[4],
            "year": r[5],
            "genre": r[6]
        } for r in rows
    ]

@app.post("/process-url")
async def process_url(payload: dict):
    url = payload.get("url")
    title = payload.get("title", "unknown")

    description = payload.get("description", "")
    youtube_id = payload.get("youtube_id", "")
    year = payload.get("year", "2024")
    genre = payload.get("genre", "Trailer")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "video.mp4")
        try:
            subprocess.run([
                "yt-dlp",
                "--js-runtimes", "node",
                "-f", "bestaudio+bestvideo/best",
                "--merge-output-format", "mp4",
                "-o", out_path,
                url
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {e.stderr.decode()}")

        # Reutilizar el pipeline normal
        object_name = f"{uuid.uuid4()}.mp4"
        minio_path = upload_file(out_path, object_name)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO videos 
                       (filename, minio_path, status, youtube_url, youtube_id, description, release_year, genre) 
                       VALUES (%s, %s, 'transcribing', %s, %s, %s, %s, %s) RETURNING id""",
                    (title, minio_path, url, youtube_id, description, str(year), genre),
                )
                video_id = cur.fetchone()[0]
            conn.commit()

        segments = transcribe(out_path)

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
                        extract_frame(out_path, mid, tmp_frame_path)
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

        model = index_segments(saved_segments, video_id)
        index_hierarchical(saved_segments, video_id, model)

        # Limpia la memoria VRAM residual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JSONResponse({
            "video_id": video_id,
            "title": title,
            "segments": len(saved_segments),
        })

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/metadata")
async def get_metadata(payload: dict):
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(payload["url"], download=False)
        return {"title": info.get("title", ""), "description": info.get("description", "")}