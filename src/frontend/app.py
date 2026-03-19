# src/frontend/app.py
import streamlit as st
import requests
import base64
from minio import Minio
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://transcription:8001")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

def get_minio():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

def load_frame(frame_path: str) -> Image.Image | None:
    if not frame_path:
        return None
    try:
        bucket, obj = frame_path.split("/", 1)
        client = get_minio()
        response = client.get_object(bucket, obj)
        return Image.open(io.BytesIO(response.read()))
    except Exception:
        return None

# ── Layout ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video RAG", page_icon="🎬", layout="wide")
st.title("🎬 Video RAG")

tab_upload, tab_segments, tab_chat = st.tabs(["📤 Upload", "🎞️ Segments", "💬 Chat"])

# ── TAB 1: Upload ────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload a video")
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "mkv", "avi", "mov"])

    if uploaded and st.button("Upload & Process", type="primary"):
        with st.spinner("Uploading, transcribing and indexing... this may take a while ⏳"):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                timeout=600,
            )
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Done! Video ID: **{data['video_id']}** — {data['segments']} segments")
            st.session_state["video_id"] = data["video_id"]
            st.json(data["preview"])
        else:
            st.error(f"Error: {response.text}")

# ── TAB 2: Segments ──────────────────────────────────────────────────────────
with tab_segments:
    st.header("Video segments")

    video_id_input = st.number_input("Video ID", min_value=1, step=1,
                                      value=st.session_state.get("video_id", 1))

    if st.button("Load segments"):
        with st.spinner("Loading..."):
            r = requests.get(f"{API_URL}/videos/{video_id_input}/segments", timeout=30)

        if r.status_code == 200:
            segments = r.json()
            st.session_state["segments"] = segments
            st.session_state["segments_video_id"] = video_id_input
        else:
            st.error("Could not load segments")

    if "segments" in st.session_state:
        segments = st.session_state["segments"]
        st.write(f"**{len(segments)} segments** for video {st.session_state.get('segments_video_id')}")

        for seg in segments:
            with st.expander(f"⏱ {seg['start']}s – {seg['end']}s | {seg['text']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    img = load_frame(seg.get("frame_path"))
                    if img:
                        st.image(img, caption=f"Frame at {round((seg['start']+seg['end'])/2, 2)}s")
                    else:
                        st.write("No frame available")
                with col2:
                    st.markdown(f"**🗣 Spoken:** {seg['text']}")
                    if seg.get("scene_desc"):
                        st.markdown(f"**👁 Visual:** {seg['scene_desc']}")
                    st.markdown(f"**🕒 Timestamps:** `{seg['start']}s → {seg['end']}s`")

# ── TAB 3: Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask about the video")

    video_id_chat = st.number_input("Video ID to query", min_value=1, step=1,
                                     value=st.session_state.get("video_id", 1),
                                     key="chat_video_id")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Mostrar historial
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask something about the video..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                r = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "video_id": int(video_id_chat)},
                    timeout=120,
                )
            if r.status_code == 200:
                data = r.json()
                answer = data["answer"]
                sources = data["sources"]

                st.markdown(answer)

                with st.expander("📎 Sources"):
                    for src in sources:
                        st.markdown(f"- **[{src['start']}s – {src['end']}s]** {src['text']} *(score: {src['score']})*")

                st.session_state["messages"].append({"role": "assistant", "content": answer})
            else:
                st.error(f"Error: {r.text}")