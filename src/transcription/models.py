# src/transcription/models.py
CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS videos (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    minio_path  TEXT NOT NULL,
    duration_s  FLOAT,
    status      TEXT DEFAULT 'pending',  -- pending | transcribed | indexed | error
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS segments (
    id            SERIAL PRIMARY KEY,
    video_id      INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    start_s       FLOAT NOT NULL,
    end_s         FLOAT NOT NULL,
    text          TEXT NOT NULL,
    frame_path    TEXT,       -- se rellena en el paso siguiente
    scene_desc    TEXT,       -- se rellena en el paso siguiente
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_start  ON segments(video_id, start_s);
"""