# app.py
import io
import os
import time
import zipfile
import streamlit as st
from typing import List, Dict
from converter import convert_to_bundle
from parser import parse_code

st.set_page_config(page_title="AI Mainframe Modernizer", page_icon="🚀", layout="wide")

st.title("🚀 AI Mainframe Modernizer — Enterprise Demo")
st.caption("Upload COBOL/JCL → get production-grade modern artifacts: code, tests, API spec, CI pipeline, Dockerfile & migration notes.")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Groq"], help="Choose which LLM backend to use")

    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], help="4o = higher quality, 4o-mini = cheaper/faster")
        key_hint = "🔐 Reads OpenAI key from env var **MY_NEW_APP_KEY**"
        missing = not os.getenv("MY_NEW_APP_KEY")
    else:
        # Common fast Groq models; pick what you’ve enabled in your Groq console
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        key_hint = "🔐 Reads Groq key from env var **GROQ_API_KEY**"
        missing = not os.getenv("GROQ_API_KEY")

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max output tokens", min_value=512, max_value=8192, value=4096, step=256)

    st.divider()
    target_stack = st.selectbox(
        "Target stack",
        ["Java (Spring Boot)", "Python (FastAPI)", "C# (.NET)", "Node.js (Express)"],
    )
    extras = st.multiselect(
        "Artifacts to include",
        ["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "K8s Manifests", "Migration Notes"],
        default=["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "Migration Notes"]
    )
    st.divider()
    st.caption(key_hint)
    if missing:
        st.warning("No API key detected for the selected provider. Set it in your environment or Streamlit secrets.")

st.write("### 1) Upload legacy file")
uploaded = st.file_uploader("Supported: .cbl, .cob, .jcl, .txt", type=["cbl", "cob", "jcl", "txt"])

st.write("### 2) Modernization instructions")
default_prompt = (
    "Convert to clean, production-ready code using the selected stack. "
    "Preserve business logic, remove dead code, and use best practices. "
    "If data structures are implicit, make them explicit. Provide comments where intent is unclear."
)
user_prompt = st.text_area("Prompt", value=default_prompt, height=140)

run = st.button("🛠️ Generate Modernization Bundle", type="primary", use_container_width=True)

if run:
    if not uploaded:
        st.error("Please upload a legacy file.")
        st.stop()
    if not user_prompt.strip():
        st.error("Please enter your instructions.")
        st.stop()

    raw = uploaded.getvalue()
    try:
        legacy_text = raw.decode("utf-8", errors="replace")
    except Exception:
        st.error("Could not read file as UTF-8 text. Please provide a text file.")
        st.stop()

    with st.status("Parsing legacy code…", expanded=False) as s:
        structured = parse_code(legacy_text, file_name=uploaded.name)
        time.sleep(0.2)
        s.update(label="Calling the model…")

    try:
        bundle = convert_to_bundle(
            legacy_code=structured,
            target_stack=target_stack,
            instructions=user_prompt,
            requested_artifacts=extras,
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            provider=provider,  # NEW
        )
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        st.stop()

    st.success("✅ Bundle ready!")

    # bundle format:
    # {
    #   "files": [{"path": "src/Main.java", "content": "…"}, ...],
    #   "notes_markdown": "…",
    #   "usage": {"provider": "...", "model": "...", ...}
    # }

    files: List[Dict[str, str]] = bundle.get("files", [])
    notes = bundle.get("notes_markdown", "")
    usage = bundle.get("usage", {})

    tab_files, tab_notes, tab_usage = st.tabs(["📦 Files", "📝 Notes", "📊 Usage"])

    def _guess_lang(path: str) -> str:
        p = path.lower()
        if "dockerfile" in p: return "docker"
        ext = os.path.splitext(p)[1].lstrip(".")
        return {
            "java": "java", "xml": "xml",
            "py": "python",
            "cs": "csharp",
            "js": "javascript", "mjs": "javascript", "cjs": "javascript",
            "yml": "yaml", "yaml": "yaml",
            "json": "json",
            "md": "md",
        }.get(ext, "text")

    with tab_files:
        if not files:
            st.info("No files returned.")
        else:
            st.write("#### Generated files")
            for f in files:
                path = f.get("path", "output.txt")
                content = f.get("content", "")
                with st.expander(path, expanded=True):
                    st.code(content, language=_guess_lang(path))

            # ZIP download
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.writestr(f.get("path", "output.txt"), f.get("content", ""))
                if notes:
                    zf.writestr("MIGRATION_NOTES.md", notes)
            mem.seek(0)
            st.download_button("⬇️ Download ZIP", data=mem, file_name="modernization_bundle.zip", mime="application/zip")

    with tab_notes:
        if notes:
            st.markdown(notes)
        else:
            st.info("No notes were returned. Enable **Migration Notes** in the sidebar.")

    with tab_usage:
        st.write("Token/provider usage reported by the API (if available).")
        st.json(usage or {"info": "No usage available from provider"})

