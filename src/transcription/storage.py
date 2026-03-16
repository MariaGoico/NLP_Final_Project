# src/transcription/storage.py
import os
from minio import Minio

BUCKET = "videos"

def get_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

def ensure_bucket():
    client = get_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

def upload_file(local_path: str, object_name: str) -> str:
    client = get_client()
    ensure_bucket()
    client.fput_object(BUCKET, object_name, local_path)
    return f"{BUCKET}/{object_name}"