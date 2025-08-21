# app.py
import io
import os
import difflib
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from converter import convert_to_bundle, chatbot, detect_legacy_type
from parser import parse_code  # keep your existing lightweight normalizer

# IMPORTANT:
# Do NOT set server.maxUploadSize here (Render blocks on-the-fly changes).
# Configure it in /.streamlit/config.toml instead.

st.set_page_config(page_title="AI Mainframe Modernizer", page_icon="🚀", layout="wide")

st.title("🚀 AI Mainframe Modernizer — Enterprise Demo")
st.caption(
    "Upload COBOL / JCL / Copybooks / Control Cards (files or zipped folders). "
    "Get modern artifacts (code, tests, OpenAPI, CI pipeline, Dockerfile) + migration notes and missing inputs."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Groq"])

    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        key_hint = "🔐 Reads OpenAI key from env var *MY_NEW_APP_KEY*"
        missing = not os.getenv("MY_NEW_APP_KEY")
    else:
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        key_hint = "🔐 Reads Groq key from env var *GROQ_API_KEY*"
        missing = not os.getenv("GROQ_API_KEY")

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max output tokens", min_value=512, max_value=8192, value=4096, step=256)

    st.divider()
    target_stack = st.selectbox(
        "Target stack",
        ["Java (Spring Boot)", "Python (FastAPI)", "C# (.NET)", "Node.js (Express)", "Groovy (for JCL)"],
    )
    extras = st.multiselect(
        "Artifacts to include",
        ["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "K8s Manifests", "Migration Notes"],
        default=["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "Migration Notes"]
    )
    st.caption(key_hint)
    if missing:
        st.warning("⚠️ No API key detected for the selected provider.")

# ---------------- Upload + Prompt ----------------
st.write("### 1) Upload files or a ZIP folder")
# Accept everything; we will auto-detect content & skip binary
uploads = st.file_uploader(
    "You can drag entire folders zipped (.zip). Text files with or without extensions are supported.",
    type=None,
    accept_multiple_files=True,
)

st.write("### 2) Modernization instructions")
default_prompt = (
    "Convert to production-ready code using the selected stack. "
    "Preserve business logic, remove dead code, and use best practices. "
    "If data structures are implicit, make them explicit. Provide comments where intent is unclear."
)
user_prompt = st.text_area("Prompt", value=default_prompt, height=140)

run = st.button("🛠️ Generate Modernization Bundle", type="primary", use_container_width=True)

# ---------- helpers ----------
def _guess_lang(path: str) -> str:
    p = path.lower()
    if "dockerfile" in p: return "docker"
    ext = Path(p).suffix.lstrip(".")
    return {
        "java": "java", "xml": "xml",
        "py": "python",
        "cs": "csharp",
        "js": "javascript", "mjs": "javascript", "cjs": "javascript",
        "groovy": "groovy",
        "yml": "yaml", "yaml": "yaml",
        "json": "json",
        "md": "md",
        "sh": "bash",
    }.get(ext, "text")


def _read_zip(file) -> List[Tuple[str, str]]:
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(file.getvalue()), "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                try:
                    raw = zf.read(info.filename)
                except Exception:
                    continue
                # Try decode as text
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = raw.decode("latin-1")
                    except Exception:
                        continue
                out.append((info.filename, text))
    except zipfile.BadZipFile:
        # Not a valid zip; ignore
        pass
    return out


def _read_uploads(files) -> List[Tuple[str, str]]:
    sources = []
    for f in files or []:
        name = f.name or "uploaded"
        if name.lower().endswith(".zip"):
            sources.extend(_read_zip(f))
            continue
        raw = f.getvalue()
        # Try decode as text (support files without extension)
        text = None
        for enc in ("utf-8", "latin-1", "utf-16"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            # skip binary files
            continue
        sources.append((name, text))
    return sources


# ---------------- Execution ----------------
if run:
    if not uploads:
        st.error("Please upload at least one file (or a ZIP).")
        st.stop()
    if not user_prompt.strip():
        st.error("Please enter your instructions.")
        st.stop()

    sources = _read_uploads(uploads)
    if not sources:
        st.error("No readable text files found in the upload.")
        st.stop()

    # Show detection results up-front
    with st.expander("🔎 Detected file types (from content)", expanded=True):
        for name, text in sources:
            st.write(f"- *{name}* → {detect_legacy_type(text)}")

    all_files: List[Dict[str, str]] = []
    all_notes: List[str] = []
    missing_refs: List[str] = []
    usage_merged: Dict[str, str] = {}

    with st.status("Converting…", expanded=False) as s:
        for idx, (fname, content) in enumerate(sources, start=1):
            s.update(label=f"Processing {fname} ({idx}/{len(sources)})")
            structured = parse_code(content, file_name=fname)  # keep your existing light parser

            try:
                bundle = convert_to_bundle(
                    legacy_code=structured,
                    target_stack=target_stack,
                    instructions=user_prompt,
                    requested_artifacts=extras,
                    model=model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    provider=provider,
                    extra_context={"source_filename": fname},
                    # enable_chunking=False  # keep default off
                )
            except Exception as e:
                st.error(f"Conversion failed for {fname}: {e}")
                continue

            all_files.extend(bundle.get("files", []))
            if bundle.get("notes"):
                all_notes.append(f"### Notes for *{fname}*\n{bundle['notes']}")
            missing_refs.extend(bundle.get("missing", []))
            usage_merged = {**usage_merged, **bundle.get("usage", {})}

    st.success("✅ Conversion complete")

    # ---------------- Tabs ----------------
    tab_files, tab_diff, tab_notes, tab_missing, tab_usage, tab_chat = st.tabs(
        ["📦 Files", "🆚 Comparison", "📝 Notes", "⚠️ Missing", "📊 Usage", "💬 Chatbot"]
    )

    # ---------- Files ----------
    with tab_files:
        if not all_files:
            st.info("No files returned.")
        else:
            st.write("#### Generated files")
            for f in all_files:
                path = f.get("path", "output.txt")
                content = f.get("content", "")
                with st.expander(path, expanded=True):
                    st.code(content, language=_guess_lang(path))

            # ZIP download
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in all_files:
                    zf.writestr(f.get("path", "output.txt"), f.get("content", ""))
                if all_notes:
                    zf.writestr("MIGRATION_NOTES.md", "\n\n---\n\n".join(all_notes))
                if missing_refs:
                    zf.writestr("MISSING_REFERENCES.md", "\n".join(f"- {m}" for m in missing_refs))
            mem.seek(0)
            st.download_button("⬇️ Download ZIP", data=mem, file_name="modernization_bundle.zip", mime="application/zip")

    # ---------- Comparison ----------
    with tab_diff:
        st.write("#### Legacy vs Modernized")
        if not all_files:
            st.info("Generate a bundle to see comparison.")
        else:
            # Choose a modern file to compare
            modern_paths = [f["path"] for f in all_files]
            chosen = st.selectbox("Choose a modernized file", modern_paths)
            modern_content = next((f["content"] for f in all_files if f["path"] == chosen), "")

            # Try to map to a likely legacy file by base name; otherwise let user pick
            base = Path(chosen).stem.lower()
            legacy_candidates = [name for name, _ in sources]
            guess = next((name for name, _ in sources if Path(name).stem.lower() == base), legacy_candidates[0])
            legacy_name = st.selectbox("Choose legacy source", legacy_candidates, index=legacy_candidates.index(guess))
            legacy_content = next(text for name, text in sources if name == legacy_name)

            diff = difflib.HtmlDiff().make_table(
                legacy_content.splitlines(), modern_content.splitlines(),
                fromdesc=f"Legacy ({legacy_name})", todesc=chosen, context=True, numlines=5
            )
            st.components.v1.html(diff, height=650, scrolling=True)

    # ---------- Notes ----------
    with tab_notes:
        if all_notes:
            st.markdown("\n\n---\n\n".join(all_notes))
        else:
            st.info("No notes were returned. Enable *Migration Notes* in the sidebar.")

    # ---------- Missing ----------
    with tab_missing:
        if missing_refs:
            st.warning("*Detected missing/dependent inputs:*\n\n" + "\n".join(f"- {m}" for m in missing_refs))
        else:
            st.success("No obvious missing inputs detected.")

    # ---------- Usage ----------
    with tab_usage:
        st.write("Token/provider usage (if available).")
        st.json(usage_merged or {"info": "No usage available"})

    # ---------- Chatbot ----------
    with tab_chat:
        st.write("#### Ask questions about your code/migration")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        if prompt_q := st.chat_input("Ask me anything about modernization…"):
            st.session_state.chat_history.append(("user", prompt_q))
            with st.chat_message("user"):
                st.markdown(prompt_q)

            legacy_all = "\n\n---\n\n".join(text for _, text in sources)
            with st.chat_message("assistant"):
                try:
                    answer = chatbot(
                        legacy_code=legacy_all,
                        query=prompt_q,
                        model=model,
                        temperature=float(temperature),
                        provider=provider,
                    )
                except Exception as e:
                    answer = f"Error: {e}"
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))
