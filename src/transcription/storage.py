# src/transcription/storage.py
import os
from minio import Minio

BUCKET = "videos"
FRAMES_BUCKET = "frames"

def get_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

def ensure_bucket(bucket: str = BUCKET):
    client = get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

def upload_file(local_path: str, object_name: str) -> str:
    client = get_client()
    ensure_bucket(BUCKET)
    client.fput_object(BUCKET, object_name, local_path)
    return f"{BUCKET}/{object_name}"

def upload_frame(local_path: str, object_name: str) -> str:
    client = get_client()
    ensure_bucket(FRAMES_BUCKET)
    client.fput_object(FRAMES_BUCKET, object_name, local_path, content_type="image/jpeg")
    return f"{FRAMES_BUCKET}/{object_name}"