import os
import sys
from pathlib import Path
#  ./.venv/bin/streamlit run ui/streamlit_app.py
# uv run --offline --no-sync python -m streamlit run ui/streamlit_app.py
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ChromaDBConfig, EmbeddingConfig
from src.retrival_pipeline import RetrievalPipeline

BASELINE_PROFILE = "Pretrained BGE (baseline)"
FINETUNED_PROFILE = "Fine-tuned BGE (synthetic trial)"
FINETUNING_DIRECTORY = PROJECT_ROOT / "src" / "finetuning"
FINETUNING_ARTIFACTS = FINETUNING_DIRECTORY / "artifacts"
FINETUNED_CHECKPOINT = FINETUNING_ARTIFACTS / "checkpoints" / "pilot_v1"
FINETUNED_PERSIST_DIRECTORY = FINETUNING_ARTIFACTS / "chroma_finetuned_synthetic_v1"
FINETUNED_MANIFEST = FINETUNING_ARTIFACTS / "manifest.v1.jsonl"
FINETUNED_PROFILE_AVAILABLE = all(
    path.is_file()
    for path in (
        FINETUNING_DIRECTORY / "common.py",
        FINETUNING_DIRECTORY / "index.py",
        FINETUNED_CHECKPOINT / "config.json",
        FINETUNED_CHECKPOINT / "model.safetensors",
        FINETUNED_PERSIST_DIRECTORY / "chroma.sqlite3",
        FINETUNED_MANIFEST,
    )
)


st.set_page_config(
    page_title="Gator Policy Guide",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --uf-blue: #0021A5;
        --uf-orange: #FA4616;
        --ink: #17243b;
        --muted: #64748b;
        --paper: #f5f7fb;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 68% 0%, rgba(250, 70, 22, .09), transparent 34rem),
            linear-gradient(180deg, #f8faff 0%, var(--paper) 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062879 0%, #00174d 100%);
    }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
        border: 1px solid #ffb49e !important;
        background-color: #fa4616 !important;
        border-radius: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"]:hover {
        background-color: #d9380f !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"] {
        background-color: transparent !important;
        color: #fff !important;
        caret-color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] {
        background-color: transparent !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] svg {
        fill: currentColor !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"]:focus-within {
        border-color: #fff !important;
        box-shadow: 0 0 0 2px #ffb49e !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border: 1px solid #ffb49e !important;
        border-radius: 12px;
        background-color: #fff1e8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlertContainer"],
    [data-testid="stSidebar"] [data-testid^="stAlertContent"] {
        background-color: #fff1e8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #17243b !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] svg {
        color: #fa4616 !important;
        fill: currentColor !important;
    }
    [data-baseweb="popover"] [role="listbox"] { background: #fff; }
    [data-baseweb="popover"] [role="option"] { color: #17243b; }
    [data-baseweb="popover"] [role="option"]:hover {
        background: #fff1e8;
        color: #17243b;
    }
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: #e8efff;
        color: #0021a5;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #cbd5e1; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        border: 1px solid rgba(255,255,255,.7);
        border-radius: 26px;
        background: linear-gradient(118deg, #062879 0%, #0038a8 62%, #1454d1 100%);
        box-shadow: 0 20px 55px rgba(0, 33, 165, .16);
        color: #fff;
    }
    .hero:after {
        content: '🐊';
        position: absolute;
        right: 3rem;
        top: 50%;
        transform: translateY(-53%) rotate(-9deg);
        font-size: 7rem;
        opacity: .14;
        filter: grayscale(1) brightness(0) invert(1);
    }
    .hero-kicker {
        color: #ffb49e;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .17em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .45rem 0 .45rem;
        color: #fff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 5vw, 3.15rem);
        font-weight: 800;
        letter-spacing: -.045em;
    }
    .hero p {
        max-width: 640px;
        margin: 0;
        color: #e2eafe;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .section-label {
        margin: 1.8rem 0 .7rem;
        color: var(--muted);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid #e4eaf3;
        border-radius: 18px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 8px 26px rgba(20, 42, 84, .045);
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #eef3ff;
    }
    .stButton > button {
        border: 1px solid #dce5f2;
        border-radius: 13px;
        background: rgba(255,255,255,.82);
        color: var(--uf-blue);
        font-weight: 700;
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }
    .stButton > button:hover {
        border-color: #9eb4ee;
        box-shadow: 0 7px 18px rgba(0, 33, 165, .1);
        color: var(--uf-blue);
        transform: translateY(-1px);
    }
    .stForm {
        padding: 1rem 1.1rem .7rem;
        border: 1px solid #e2e9f4;
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 10px 32px rgba(20, 42, 84, .065);
    }
    .stLinkButton > a {
        border-radius: 11px !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] {
        border-color: #e5eaf2;
        border-radius: 13px;
        background: rgba(248,250,255,.75);
    }
    .trust-note {
        padding: .85rem 1rem;
        border-left: 3px solid var(--uf-orange);
        border-radius: 0 12px 12px 0;
        background: rgba(255,255,255,.72);
        color: #526179;
        font-size: .86rem;
        line-height: 1.5;
    }
    @media (max-width: 700px) {
        .hero { padding: 1.45rem; border-radius: 20px; }
        .hero:after { right: 1rem; font-size: 4rem; }
        .hero p { max-width: 80%; font-size: .94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">University of Florida · Policy, made clearer</div>
        <h1>Ask the Gator.</h1>
        <p>Get a clear answer grounded in UF policy—and open the original policy page to verify it for yourself.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Warming up the Gator…")
def load_pipeline(
    model_name: str,
    persist_directory: str,
    collection_name: str,
    checkpoint_path: str | None = None,
) -> RetrievalPipeline:
    persist_path = Path(persist_directory)
    if not persist_path.is_dir():
        raise FileNotFoundError(f"Chroma index directory not found: {persist_path}")

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint is not None:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint}")
        if not FINETUNED_MANIFEST.is_file():
            raise FileNotFoundError(f"Fine-tuning manifest not found: {FINETUNED_MANIFEST}")

        from src.finetuning.common import sha256_file
        from src.finetuning.index import compatible

    pipeline = RetrievalPipeline(
        embedding_config=EmbeddingConfig(model_name=model_name),
        chroma_config=ChromaDBConfig(
            persist_directory=str(persist_path),
            collection_name=collection_name,
            require_existing_collection=True,
        ),
    )
    pipeline.initialize()

    collection = pipeline.chroma_manager.collection
    if collection is None or collection.count() == 0:
        raise ValueError(f"Chroma collection is empty: {collection_name}")

    if checkpoint is not None:
        metadata = collection.metadata or {}
        if (
            metadata.get("evaluation_type") != "assistant_reviewed_synthetic_questions"
            or metadata.get("independent_human_test") is not False
        ):
            raise ValueError("Selected fine-tuned index is not labeled as a synthetic trial")
        compatible(
            metadata,
            checkpoint,
            pipeline.embedding_model.get_dimension(),
            sha256_file(FINETUNED_MANIFEST),
        )
    return pipeline


profile_options = [BASELINE_PROFILE]
if FINETUNED_PROFILE_AVAILABLE:
    profile_options.append(FINETUNED_PROFILE)

if st.session_state.get("retrieval_profile") not in profile_options:
    st.session_state["retrieval_profile"] = BASELINE_PROFILE

with st.sidebar:
    selected_profile = st.selectbox(
        "Retrieval model and index",
        options=profile_options,
        key="retrieval_profile",
        help="The model and Chroma collection are switched together so their vectors stay compatible.",
    )
    if selected_profile == FINETUNED_PROFILE:
        st.warning(
            "Experimental synthetic-trial model. Its evaluation used AI-generated "
            "questions and is not an independent human test."
        )
    else:
        st.caption("Original pretrained BGE model and baseline Chroma collection.")
        if not FINETUNED_PROFILE_AVAILABLE:
            st.info("The fine-tuned profile is unavailable because its local code or model/index artifacts are missing.")

if st.session_state.get("_active_retrieval_profile") not in (None, selected_profile):
    st.session_state["messages"] = []
    st.sidebar.info("Conversation cleared after switching retrieval models.")
st.session_state["_active_retrieval_profile"] = selected_profile

if not os.environ.get("OPENAI_API_KEY"):
    st.warning(
        "Add your API key to the project-root `.env` file as "
        "`OPENAI_API_KEY=your-key`, then restart Streamlit."
    )
    st.stop()

if selected_profile == FINETUNED_PROFILE:
    profile = {
        "model_name": str(FINETUNED_CHECKPOINT),
        "persist_directory": str(FINETUNED_PERSIST_DIRECTORY),
        "collection_name": "policies_finetuned_synthetic",
        "checkpoint_path": str(FINETUNED_CHECKPOINT),
    }
else:
    profile = {
        "model_name": EmbeddingConfig().model_name,
        "persist_directory": str(PROJECT_ROOT / "data" / "chroma_db"),
        "collection_name": "policies",
        "checkpoint_path": None,
    }

try:
    pipeline = load_pipeline(**profile)
except (OSError, RuntimeError, ValueError) as exc:
    st.error(f"Could not load {selected_profile}: {exc}")
    st.stop()

stats = pipeline.chroma_manager.get_collection_stats()

with st.sidebar:
    st.markdown("## 🐊 Gator Guide")
    st.caption("Your friendly guide to University of Florida policies.")
    st.divider()

    st.markdown("### Search settings")
    n_results = st.slider(
        "Policy sources",
        min_value=1,
        max_value=6,
        value=3,
        help="How many relevant policy passages to retrieve and cite.",
    )
    st.metric("Indexed policy passages", f"{stats['count']:,}")
    st.caption(f"Collection: `{stats['name']}`")
    st.divider()
    st.markdown("### Check the source")
    st.markdown(
        "Every answer comes with matching passages and a direct link to the "
        "official UF policy page. Open a citation to verify the wording."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown('<div class="section-label">A few things you can ask</div>', unsafe_allow_html=True)
suggestions = [
    "How do I request paid medical leave?",
    "What is UF's policy on remote work?",
    "How can I report a workplace concern?",
]
suggestion_columns = st.columns(len(suggestions))
for column, suggestion in zip(suggestion_columns, suggestions):
    with column:
        if st.button(suggestion, key=f"suggestion-{suggestion}", use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No source passages were returned for this answer.")
        return

    st.markdown("**Sources · open the official page to verify**")
    grouped_sources: list[dict] = []
    group_indexes: dict[tuple[str, str | int], int] = {}
    for passage_number, source in enumerate(sources, start=1):
        link = source.get("link")
        group_key = ("link", link) if link else ("passage", passage_number)
        if group_key not in group_indexes:
            group_indexes[group_key] = len(grouped_sources)
            grouped_sources.append({"source": source, "passages": []})
        grouped_sources[group_indexes[group_key]]["passages"].append(
            (passage_number, source)
        )

    for group in grouped_sources:
        source = group["source"]
        title = source.get("title") or "UF policy"
        category = source.get("category") or "University of Florida"
        citation_numbers = ", ".join(
            f"[{number}]" for number, _ in group["passages"]
        )
        with st.expander(f"{citation_numbers} {title}  ·  {category}"):
            notice = next(
                (
                    passage.get("notice")
                    for _, passage in group["passages"]
                    if passage.get("notice")
                ),
                None,
            )
            if notice:
                st.warning(notice)
            for passage_number, passage in group["passages"]:
                st.caption(
                    f"Passage [{passage_number}] · "
                    f"{passage.get('type', 'Policy')}"
                )
                st.markdown(
                    passage.get("excerpt") or "_No passage excerpt available._"
                )
                distance = passage.get("distance")
                if distance is not None:
                    st.caption(
                        f"Semantic distance: {distance:.3f} "
                        "(lower is a closer match)"
                    )
            link = source.get("link")
            if link:
                st.link_button(
                    "Open official UF policy page ↗",
                    link,
                    use_container_width=True,
                )
            else:
                st.warning("An official page link could not be matched to this passage.")


def render_message(message: dict) -> None:
    with st.chat_message("assistant", avatar="🐊"):
        if message.get("error"):
            st.error(message["error"])
            return
        st.markdown(message["answer"])
        render_sources(message.get("sources", []))


for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🐊"):
            st.markdown(message["content"])
    else:
        render_message(message)

st.markdown('<div class="section-label">Your question</div>', unsafe_allow_html=True)
pending_question = st.session_state.pop("pending_question", None)
with st.form("policy-question", clear_on_submit=True):
    question_input = st.text_input(
        "Ask a question about UF policy",
        placeholder="For example: How much notice should I give before requesting leave?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Ask the Gator  →",
        type="primary",
        use_container_width=True,
    )

question = pending_question or (question_input if submitted else "")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    try:
        with st.spinner("Finding the right policy passages…"):
            result = pipeline.query(question=question, n_results=n_results)
        if result.get("error"):
            raise RuntimeError(result["error"])
        answer = result.get("answer")
        if not answer:
            raise RuntimeError("The language model returned an empty answer.")
        assistant_message = {
            "role": "assistant",
            "answer": answer,
            "sources": result.get("sources", []),
        }
    except Exception as exc:
        assistant_message = {
            "role": "assistant",
            "error": f"Could not answer this question: {exc}",
        }
    st.session_state["messages"].append(assistant_message)
    render_message(assistant_message)

with st.expander("Project screenshots", expanded=False):
    screenshot_gallery = [
        (
            "Screenshot 2026-11-11 at 4.53.48 PM.png",
            "Gator Policy Guide Streamlit interface",
        ),
        (
            "Screenshot 2026-11-11 at 4.54.14 PM.png",
            "Retrieved policy passages and citation details",
        ),
        (
            "Screenshot 2026-11-11 at 4.54.28 PM.png",
            "Official UF policy page used to verify a citation",
        ),
    ]
    for start in range(0, len(screenshot_gallery), 2):
        image_columns = st.columns(2)
        for column, (filename, caption) in zip(
            image_columns,
            screenshot_gallery[start : start + 2],
        ):
            image_path = PROJECT_ROOT / "docs" / "visuals" / filename
            if image_path.is_file():
                column.image(str(image_path), caption=caption, use_container_width=True)
            else:
                column.warning(f"Screenshot is missing: {filename}")

st.markdown(
    '<div class="trust-note">AI can make mistakes. Use the linked official UF policy pages as the source of truth.</div>',
    unsafe_allow_html=True,
)
