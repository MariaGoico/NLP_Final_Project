# src/frontend/app.py
import streamlit as st
import requests
from minio import Minio
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://transcription:8001")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

FILMS = [
    {
        "title": "The Devil Wears Prada",
        "year": 2006,
        "genre": "Comedy / Drama",
        "description": "A smart but naive college grad lands a job as an assistant to Miranda Priestly, the most powerful woman in fashion.",
        "youtube_url": "https://www.youtube.com/watch?v=6ZOZwUQKu3E",
        "youtube_id": "6ZOZwUQKu3E",
        "thumbnail": "https://img.youtube.com/vi/6ZOZwUQKu3E/hqdefault.jpg",
    },
    {
        "title": "Legally Blonde",
        "year": 2001,
        "genre": "Comedy / Romance",
        "description": "Elle Woods, a fashionable sorority queen, follows her ex-boyfriend to Harvard Law and shows everyone she's capable of more.",
        "youtube_url": "https://www.youtube.com/watch?v=vWOHwI_FgAo",
        "youtube_id": "vWOHwI_FgAo",
        "thumbnail": "https://img.youtube.com/vi/vWOHwI_FgAo/hqdefault.jpg",
    },
    {
        "title": "Harry Potter and the Sorcerer's Stone",
        "year": 2001,
        "genre": "Fantasy / Adventure",
        "description": "An orphaned boy discovers he's a wizard and begins his education at Hogwarts School of Witchcraft and Wizardry.",
        "youtube_url": "https://www.youtube.com/watch?v=VyHV0BRtdxo",
        "youtube_id": "VyHV0BRtdxo",
        "thumbnail": "https://img.youtube.com/vi/VyHV0BRtdxo/hqdefault.jpg",
    },
]

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

def get_processed_videos() -> dict:
    """Devuelve dict title → video_id de vídeos ya procesados."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code != 200:
            return {}
    except Exception:
        return {}
    return st.session_state.get("processed_videos", {})

# ── Página config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineRAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e50914, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0.5rem;
    }
    .film-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #2a2a4a;
        height: 100%;
    }
    .film-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0 0.2rem 0;
    }
    .film-meta {
        font-size: 0.8rem;
        color: #e50914;
        margin-bottom: 0.5rem;
    }
    .film-desc {
        font-size: 0.85rem;
        color: #aaa;
        line-height: 1.4;
    }
    .processed-badge {
        background: #1db954;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stChatMessage {background: #1a1a2e !important;}
    div[data-testid="stHorizontalBlock"] {gap: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Inicializar estado ───────────────────────────────────────────────────────
if "processed_videos" not in st.session_state:
    st.session_state["processed_videos"] = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "selected_film" not in st.session_state:
    st.session_state["selected_film"] = None

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🎬 CineRAG</p>
    <p class="main-subtitle">Ask anything about your favourite movie trailers</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_catalog, tab_chat, tab_segments = st.tabs(["🎥 Catalog", "💬 Ask", "🎞️ Segments"])

# ── TAB 1: Catálogo ──────────────────────────────────────────────────────────
with tab_catalog:
    st.markdown("### Browse & Process Trailers")
    st.caption("Watch the trailer, then click **Process** to index it for Q&A.")
    st.write("")

    cols = st.columns(3)
    for i, film in enumerate(FILMS):
        with cols[i]:
            is_processed = film["title"] in st.session_state["processed_videos"]

            # Thumbnail + badge
            st.markdown(
                f'<iframe width="100%" height="200" src="https://www.youtube.com/embed/{film["youtube_id"]}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                f'gyroscope; picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

            st.markdown(f'<p class="film-title">{film["title"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="film-meta">{film["year"]} · {film["genre"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="film-desc">{film["description"]}</p>', unsafe_allow_html=True)
            st.write("")

            if is_processed:
                vid_id = st.session_state["processed_videos"][film["title"]]
                st.markdown(f'<span class="processed-badge">✓ Indexed — ID {vid_id}</span>', unsafe_allow_html=True)
                st.write("")
            else:
                if st.button("⚙️ Process trailer", key=f"process_{i}", type="primary"):
                    with st.spinner(f"Downloading and processing *{film['title']}*... this may take a few minutes ⏳"):
                        try:
                            r = requests.post(
                                f"{API_URL}/process-url",
                                json={"url": film["youtube_url"], "title": film["title"]},
                                timeout=600,
                            )
                            if r.status_code == 200:
                                data = r.json()
                                st.session_state["processed_videos"][film["title"]] = data["video_id"]
                                st.success(f"✅ {film['title']} indexed! ({data['segments']} segments)")
                                st.rerun()
                            else:
                                st.error(f"Error: {r.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")


# ── TAB 2: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    processed = st.session_state["processed_videos"]

    if not processed:
        st.info("💡 Process at least one trailer from the **Catalog** tab to start asking questions.")
    else:
        st.markdown("### Ask anything about the trailers")

        # Selector de película
        film_options = {f"{title} (ID: {vid_id})": vid_id for title, vid_id in processed.items()}
        film_options["🎬 All processed trailers"] = None
        selected = st.selectbox("Select a film or search across all:", list(film_options.keys()))
        active_video_id = film_options[selected]

        st.divider()

        # Historial del chat
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the trailer... e.g. 'What is the main character like?'"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching through the trailer..."):
                    payload = {"question": prompt}
                    if active_video_id:
                        payload["video_id"] = active_video_id
                    r = requests.post(
                        f"{API_URL}/query",
                        json=payload,
                        timeout=120,
                    )
                if r.status_code == 200:
                    data = r.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    levels = data.get("retrieval_levels_used", {})

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                st.markdown(f"- **[{src['start']}s – {src['end']}s]** {src['text']} *(score: {src['score']})*")

                    level_used = levels.get("level_used", "hierarchical")
                    st.caption(f"🔍 {levels.get('videos', 0)} videos → {levels.get('windows', 0)} windows → {levels.get('segments', 0)} segments · *{level_used}*")

                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error: {r.text}")

        if st.session_state["messages"]:
            if st.button("🗑️ Clear chat"):
                st.session_state["messages"] = []
                st.rerun()

# ── TAB 3: Segments ───────────────────────────────────────────────────────────
with tab_segments:
    processed = st.session_state["processed_videos"]

    if not processed:
        st.info("💡 Process at least one trailer from the **Catalog** tab first.")
    else:
        st.markdown("### Explore indexed segments")

        film_options = {f"{title} (ID: {vid_id})": vid_id for title, vid_id in processed.items()}
        selected = st.selectbox("Select a film:", list(film_options.keys()), key="seg_select")
        active_video_id = film_options[selected]

        if st.button("Load segments", type="primary"):
            with st.spinner("Loading..."):
                r = requests.get(f"{API_URL}/videos/{active_video_id}/segments", timeout=30)
            if r.status_code == 200:
                st.session_state["segments"] = r.json()
            else:
                st.error("Could not load segments")

        if "segments" in st.session_state:
            segments = st.session_state["segments"]
            st.write(f"**{len(segments)} segments** indexed")
            st.write("")

            for seg in segments:
                with st.expander(f"⏱ {seg['start']}s – {seg['end']}s · {seg['text'][:80]}..."):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        img = load_frame(seg.get("frame_path"))
                        if img:
                            st.image(img, caption=f"Frame @ {round((seg['start']+seg['end'])/2, 2)}s")
                        else:
                            st.write("No frame")
                    with col2:
                        st.markdown(f"**🗣 Spoken:** {seg['text']}")
                        if seg.get("scene_desc"):
                            st.markdown(f"**👁 Visual:** {seg['scene_desc']}")
                        st.markdown(f"**🕒** `{seg['start']}s → {seg['end']}s`")