# src/frontend/app.py
import streamlit as st
import requests
from minio import Minio
from PIL import Image
import io
import os
import re

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
    {
        "title": "Dune: Part Two",
        "year": 2024,
        "genre": "Sci-Fi / Adventure",
        "description": "Paul Atreides unites with Chani and the Fremen while on a warpath of revenge against the conspirators who destroyed his family.",
        "youtube_url": "https://www.youtube.com/watch?v=Way9Dexny3w",
        "youtube_id": "Way9Dexny3w",
        "thumbnail": "https://img.youtube.com/vi/Way9Dexny3w/hqdefault.jpg",
    },
    {
        "title": "Deadpool & Wolverine",
        "year": 2024,
        "genre": "Action / Comedy / Sci-Fi",
        "description": "A weary Wolverine finds himself recovering from his injuries when he comes across a loudmouth Deadpool.",
        "youtube_url": "https://www.youtube.com/watch?v=73_1biulkYk",
        "youtube_id": "73_1biulkYk",
        "thumbnail": "https://img.youtube.com/vi/73_1biulkYk/hqdefault.jpg",
    },
    {
        "title": "A Quiet Place: Day One",
        "year": 2024,
        "genre": "Horror / Sci-Fi",
        "description": "Experience the day the world went silent in this prequel following a woman named Sam as she tries to survive an invasion in New York City.",
        "youtube_url": "https://www.youtube.com/watch?v=YPY7J-flzE8",
        "youtube_id": "YPY7J-flzE8",
        "thumbnail": "https://img.youtube.com/vi/YPY7J-flzE8/hqdefault.jpg",
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
    """Devuelve dict title -> video_id de vídeos ya procesados."""
    try:
        r = requests.get(f"{API_URL}/videos", timeout=5)
        if r.status_code == 200:
            return r.json() # Devuelve la lista desde el backend
    except Exception as e:
        print(f"Error fetching processed videos: {e}")
    return []


def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_video_metadata(url: str) -> dict | None:
    """Fetch video title and description from the backend /metadata endpoint."""
    try:
        r = requests.post(f"{API_URL}/metadata", json={"url": url}, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineRAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
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
    .upload-section {
        background: #0d0d1a;
        border: 2px dashed #e50914;
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem 2rem;
        margin: 2rem 0 1rem 0;
    }
    .upload-section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #fff;
        margin: 0 0 0.25rem 0;
    }
    .upload-section-sub {
        font-size: 0.85rem;
        color: #777;
        margin-bottom: 0;
    }
    div[data-testid="stHorizontalBlock"] {gap: 1.5rem;}
</style>
""", unsafe_allow_html=True)
    # .stChatMessage {background: #1a1a2e !important;}


# ── Session state ────────────────────────────────────────────────────────────
if "video_data" not in st.session_state:
    db_videos = get_processed_videos()
    st.session_state["video_data"] = db_videos
    
    # Para el chat (necesita un diccionario Título -> ID)
    st.session_state["processed_videos"] = {v["title"]: v["id"] for v in db_videos}
    
    # Filtramos para My Trailers (los que NO están en la lista fija FILMS)
    catalog_titles = [f["title"] for f in FILMS]
    st.session_state["my_trailers"] = [v for v in db_videos if v["title"] not in catalog_titles]

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "selected_film" not in st.session_state:
    st.session_state["selected_film"] = None
if "custom_preview" not in st.session_state:
    st.session_state["custom_preview"] = None  # {youtube_id, title, description, url, needs_manual?}
if "my_trailers" not in st.session_state:
    st.session_state["my_trailers"] = []

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

# ── TAB 1: Catalog ───────────────────────────────────────────────────────────
with tab_catalog:
    st.markdown("### Browse & Process Trailers")
    st.caption("Watch the trailer, then click **Process** to index it for Q&A.")
    st.write("")

    cols = st.columns(3)
    for i, film in enumerate(FILMS):
        with cols[i % 3]:
            is_processed = film["title"] in st.session_state["processed_videos"]

            st.markdown(
                f'<iframe width="100%" height="200" src="https://www.youtube.com/embed/{film["youtube_id"]}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                f'gyroscope; picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True,
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
                                json={
                                    "url": film["youtube_url"], 
                                    "title": film["title"],
                                    "description": film["description"],
                                    "youtube_id": film["youtube_id"],
                                    "year": str(film["year"]),
                                    "genre": film["genre"]
                                },
                                timeout=600,
                            )
                            if r.status_code == 200:
                                data = r.json()
                                st.session_state["processed_videos"][film["title"]] = data["video_id"]
                                st.success(f"✅ {film['title']} indexed! ({data['segments']} segments)")
                                del st.session_state["video_data"] # (Vaciamos la cache para forzar que vuelva a consultar la BD en el rerun)
                                st.session_state["custom_preview"] = None
                                st.rerun()
                            else:
                                st.error(f"Error: {r.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")

    # ── Custom YouTube URL uploader ──────────────────────────────────────────

    if st.session_state["my_trailers"]:
        st.markdown("### 🍿 My Trailers")
        st.divider()
        
        my_list = st.session_state["my_trailers"]
        COLUMNS_PER_ROW = 3
        
        for i in range(0, len(my_list), COLUMNS_PER_ROW):
            row_cols = st.columns(COLUMNS_PER_ROW)
            chunk = my_list[i : i + COLUMNS_PER_ROW]
            
            for j, film in enumerate(chunk):
                with row_cols[j]:
                    st.markdown(
                        f'<iframe width="100%" height="200" src="https://www.youtube.com/embed/{film["youtube_id"]}" '
                        f'frameborder="0" allowfullscreen></iframe>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<p class="film-title">{film["title"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="film-meta">{film["year"]} · {film["genre"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="film-desc">{film["description"]}</p>', unsafe_allow_html=True)
                    st.write("")
                    
                    # Custom trailers are already processed, so we just show the green badge
                    vid_id = st.session_state["processed_videos"].get(film["title"])
                    st.markdown(f'<span class="processed-badge">✓ Indexed — ID {vid_id}</span>', unsafe_allow_html=True)
                    st.write("")
                    
    st.markdown("""
    <div class="upload-section">
        <p class="upload-section-title">➕ Add your own trailer</p>
        <p class="upload-section-sub">Paste any YouTube URL — we'll fetch the title, description and thumbnail automatically.</p>
    </div>
    """, unsafe_allow_html=True)

    url_col, btn_col = st.columns([5, 1])
    with url_col:
        custom_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="custom_url_input",
        )
    with btn_col:
        fetch_clicked = st.button("🔍 Fetch info", type="secondary", use_container_width=True)

    if fetch_clicked:
        if not custom_url.strip():
            st.warning("Please paste a YouTube URL first.")
        else:
            video_id = extract_youtube_id(custom_url.strip())
            if not video_id:
                st.error("❌ Could not extract a valid YouTube video ID from that URL.")
            else:
                with st.spinner("Fetching video metadata..."):
                    meta = fetch_video_metadata(custom_url.strip())

                if meta:
                    st.session_state["custom_preview"] = {
                        "youtube_id": video_id,
                        "title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "url": custom_url.strip(),
                        "needs_manual": False,
                    }
                else:
                    # Metadata endpoint unavailable — show preview with manual fields
                    st.session_state["custom_preview"] = {
                        "youtube_id": video_id,
                        "title": "",
                        "description": "",
                        "url": custom_url.strip(),
                        "needs_manual": True,
                    }

    # ── Preview card ─────────────────────────────────────────────────────────
    preview = st.session_state.get("custom_preview")
    if preview:
        st.markdown("---")
        st.markdown("#### 🎬 Preview")
        left, right = st.columns([1, 2])

        with left:
            st.markdown(
                f'<iframe width="100%" height="185" '
                f'src="https://www.youtube.com/embed/{preview["youtube_id"]}" '
                f'frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
            # Also show the thumbnail image below the embed
            st.image(
                f'https://img.youtube.com/vi/{preview["youtube_id"]}/hqdefault.jpg',
                caption="Thumbnail (hqdefault)",
                use_container_width=True,
            )

        with right:
            if preview["needs_manual"]:
                st.caption("⚠️ Metadata endpoint not reachable — please fill in manually:")
                final_title = st.text_input(
                    "Title *", key="manual_title", placeholder="e.g. Interstellar Official Trailer"
                )
                final_desc = st.text_area(
                    "Description", key="manual_desc",
                    placeholder="Short description of the trailer...",
                    height=100,
                )
            else:
                final_title = st.text_input("Title", value=preview["title"], key="edit_title")
                final_desc = st.text_area(
                    "Description", value=preview["description"], key="edit_desc", height=100
                )

            st.write("")

            already_indexed = final_title and final_title.strip() in st.session_state["processed_videos"]
            if already_indexed:
                vid_id = st.session_state["processed_videos"][final_title.strip()]
                st.markdown(f'<span class="processed-badge">✓ Already indexed — ID {vid_id}</span>', unsafe_allow_html=True)
            else:
                proc_col, clear_col = st.columns([2, 1])
                with proc_col:
                    if st.button("⚙️ Process this trailer", key="process_custom", type="primary", use_container_width=True):
                        if not final_title.strip():
                            st.warning("Please provide a title before processing.")
                        else:
                            with st.spinner(f"Processing *{final_title}*... this may take a few minutes ⏳"):
                                try:
                                    r = requests.post(
                                        f"{API_URL}/process-url",
                                        json={
                                            "url": preview["url"], 
                                            "title": final_title.strip(),
                                            "description": final_desc.strip(),
                                            "youtube_id": preview["youtube_id"],
                                            "year": "Custom",
                                            "genre": "User Added"
                                        },
                                        timeout=600,
                                    )
                                    if r.status_code == 200:
                                        data = r.json()
                                        st.session_state["processed_videos"][final_title.strip()] = data["video_id"]

                                        st.session_state["my_trailers"].append({
                                            "title": final_title.strip(),
                                            "year": "Custom",
                                            "genre": "User Added",
                                            "description": final_desc.strip(),
                                            "youtube_url": preview["url"],
                                            "youtube_id": preview["youtube_id"]
                                        })

                                        st.success(f"✅ {final_title} indexed! ({data['segments']} segments)")
                                        st.session_state["custom_preview"] = None
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {r.text}")
                                except Exception as e:
                                    st.error(f"Connection error: {e}")
                with clear_col:
                    if st.button("✖ Clear", key="clear_preview", use_container_width=True):
                        st.session_state["custom_preview"] = None
                        st.rerun()


# ── TAB 2: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    processed = st.session_state["processed_videos"]

    if not processed:
        st.info("💡 Process at least one trailer from the **Catalog** tab to start asking questions.")
    else:
        st.markdown("### Ask anything about the trailers")

        film_options = {f"{title} (ID: {vid_id})": vid_id for title, vid_id in processed.items()}
        film_options["🎬 All processed trailers"] = None
        selected = st.selectbox("Select a film or search across all:", list(film_options.keys()))
        active_video_id = film_options[selected]

        st.divider()

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