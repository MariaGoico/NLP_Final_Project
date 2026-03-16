# src/transcription/db.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from .models import CREATE_TABLES

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "videorag"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLES)
        conn.commit()