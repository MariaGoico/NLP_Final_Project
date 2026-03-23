import streamlit as st
import requests
import io
import os
from minio import Minio
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://transcription:8001")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

@st.cache_resource
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

@st.cache_data(show_spinner=False)
def load_video_bytes(video_path: str) -> bytes | None:
    if not video_path:
        return None
    try:
        bucket, obj = video_path.split("/", 1)
        client = get_minio()
        response = client.get_object(bucket, obj)
        return response.read()
    except Exception as e:
        st.error(f"Error loading video bytes: {e}")
        return None

# Obtener la lista de videos del backend
def get_all_videos():
    try:
        r = requests.get(f"{API_URL}/videos", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []
    return []

# ── Layout & Modern Styling ────────────────────────────────────────────────
st.set_page_config(page_title="Video RAG", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    h1, h2, h3, .stMarkdown p, span { color: #212529 !important; }

    .stChatMessage { 
        border-radius: 12px; 
        border: 1px solid #dee2e6; 
        background-color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }

    .video-card { 
        border: 1px solid #dee2e6; 
        border-radius: 10px; 
        padding: 15px; 
        background: white; 
        margin-bottom: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Video RAG")

tab_upload, tab_segments, tab_chat, tab_library = st.tabs([
    "📤 Upload", "🎞️ Segments", "💬 Chat", "📁 Library"
])

video_list = get_all_videos()

if video_list:
    video_options = {f"ID: {v['id']} | {v['filename']}": v['id'] for v in video_list}
else:
    video_options = {}

# ── TAB 1: Upload ────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload a video")
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "mkv", "avi", "mov"])

    if uploaded and st.button("Upload & Process", type="primary"):
        with st.spinner("Uploading, transcribing and indexing... ⏳"):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                timeout=600,
            )
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Done! Video ID: **{data['video_id']}**")
            st.session_state["video_id"] = data["video_id"]
            st.rerun()
        else:
            st.error(f"Error: {response.text}")

# ── TAB 2: Segments ──────────────────────────────────────────────────────────
with tab_segments:
    st.header("Video segments")
    
    if not video_options:
        st.info("No videos found. Please upload a video first.")
    else:
        current_vid = st.session_state.get("video_id")
        default_index = 0
        if current_vid:
            for idx, vid in enumerate(video_options.values()):
                if vid == current_vid:
                    default_index = idx
                    break

        selected_vid_str = st.selectbox("Select a Video", options=list(video_options.keys()), index=default_index, key="seg_select")
        video_id_input = video_options[selected_vid_str]

        if st.button("Load segments"):
            with st.spinner("Loading..."):
                r = requests.get(f"{API_URL}/videos/{video_id_input}/segments", timeout=30)
            if r.status_code == 200:
                st.session_state["segments"] = r.json()
                st.session_state["segments_video_id"] = video_id_input
                st.session_state["video_id"] = video_id_input

        if "segments" in st.session_state and st.session_state.get("segments_video_id") == video_id_input:
            for seg in st.session_state["segments"]:
                with st.expander(f"⏱ {seg['start']}s – {seg['end']}s | {seg['text']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        img = load_frame(seg.get("frame_path"))
                        if img: st.image(img)
                    with col2:
                        st.markdown(f"**🗣 Spoken:** {seg['text']}")
                        if seg.get("scene_desc"):
                            st.markdown(f"**👁 Visual:** {seg['scene_desc']}")

# ── TAB 3: Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask about the video")
    
    if not video_options:
        st.info("No videos found. Please upload a video first.")
    else:
        current_vid_chat = st.session_state.get("video_id")
        default_index_chat = 0
        if current_vid_chat:
            for idx, vid in enumerate(video_options.values()):
                if vid == current_vid_chat:
                    default_index_chat = idx
                    break

        selected_vid_str_chat = st.selectbox("Select a Video to query", options=list(video_options.keys()), index=default_index_chat, key="chat_select")
        video_id_chat = video_options[selected_vid_str_chat]
        
        # Resetear el chat si cambias de video
        if video_id_chat != st.session_state.get("video_id"):
            st.session_state["video_id"] = video_id_chat
            st.session_state["messages"] = []
        
        if "messages" not in st.session_state: 
            st.session_state["messages"] = []

        # Mostrar historial (incluyendo sources y levels guardados)
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📎 Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"- **[{src['start']}s – {src['end']}s]** {src['text']} *(score: {src['score']})*")
                if "levels" in msg and msg["levels"]:
                    levels = msg["levels"]
                    st.caption(f"🔍 Retrieval: {levels['videos']} videos → {levels['windows']} windows → {levels['segments']} segments")

        # Input
        if prompt := st.chat_input("Ask something about the video..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"): 
                st.markdown(prompt)

            # Contenedor del asistente con el spinner nativo
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    r = requests.post(f"{API_URL}/query", json={"question": prompt, "video_id": int(video_id_chat)}, timeout=120)
                
                if r.status_code == 200:
                    data = r.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    levels = data.get("retrieval_levels_used")

                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                st.markdown(f"- **[{src['start']}s – {src['end']}s]** {src['text']} *(score: {src['score']})*")
                    
                    if levels:
                        st.caption(f"🔍 Retrieval: {levels['videos']} videos → {levels['windows']} windows → {levels['segments']} segments")

                    # Guardar en el historial la respuesta, las fuentes y los niveles
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "levels": levels
                    })
                else:
                    st.error(f"Error: {r.text}")

# ── TAB 4: Library ───────────────────────────────────────────────────────────
with tab_library:
    st.header("Video Library")
    
    if st.button("🔄 Refresh Library"):
        st.rerun() 
        
    if not video_list:
        st.info("The library is currently empty. Upload a video to get started!")
    else:
        for video in video_list:
            clean_date = video['created_at'].split('.')[0].replace('T', ' ')
            
            with st.container():
                st.markdown(f"""
                    <div class="video-card">
                        <h3 style="margin:0px;">ID: {video['id']} | {video['filename']}</h3>
                        <p style="color: #6c757d; font-size: 0.85rem; margin: 5px 0px;">Created: {clean_date}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                col_video, col_empty = st.columns([1, 1])
                
                with col_video:
                    with st.spinner(f"Loading video {video['id']}..."):
                        video_bytes = load_video_bytes(video['minio_path'])
                        if video_bytes:
                            st.video(video_bytes, format="video/mp4")
                        else:
                            st.error("Could not load video file from storage.")
                
                st.divider()