# app.py
import io
import os
import difflib
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from converter import convert_to_bundle, chatbot, detect_legacy_type
from parser import parse_code

# ---------------- Page ----------------
st.set_page_config(page_title="AI Mainframe Modernizer", page_icon="🚀", layout="wide")
st.title("🚀 AI Mainframe Modernizer — Enterprise Demo")
st.caption(
    "Upload COBOL/JCL/Copybooks/Control Cards (files or zipped folders) → "
    "get modern artifacts: code, tests, API spec, CI pipeline, Dockerfile & migration notes."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Groq"])

    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        key_hint = "🔐 Reads OpenAI key from env var *MY_NEW_APP_KEY*"
        missing_key = not os.getenv("MY_NEW_APP_KEY")
    else:
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        key_hint = "🔐 Reads Groq key from env var *GROQ_API_KEY*"
        missing_key = not os.getenv("GROQ_API_KEY")

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
        default=["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "Migration Notes"],
    )
    st.caption(key_hint)
    if missing_key:
        st.warning("⚠️ No API key detected for the selected provider.")

# ---------------- Upload + Prompt ----------------
st.write("### 1) Upload files or a ZIP folder")
# Accept any file; we’ll filter to text/zip in code so extensionless files are supported
uploads = st.file_uploader(
    "Supported: .cbl, .cob, .cpy, .jcl, .txt, .zip (files without extension are also accepted)",
    type=None,
    accept_multiple_files=True,
)

st.write("### 2) Modernization instructions")
default_prompt = (
    "Convert to production-ready code using the selected stack. "
    "Preserve business logic, remove dead code, and use best practices. "
    'If data structures are implicit, make them explicit. Provide comments where intent is unclear.'
)
user_prompt = st.text_area("Prompt", value=default_prompt, height=140)

run = st.button("🛠️ Generate Modernization Bundle", type="primary", use_container_width=True)

# ---------------- Helpers ----------------
TEXT_LANG_BY_EXT = {
    "java": "java", "xml": "xml",
    "py": "python",
    "cs": "csharp",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "groovy": "groovy",
    "yml": "yaml", "yaml": "yaml",
    "json": "json",
    "md": "md",
    "sh": "bash",
}
def _guess_lang(path: str) -> str:
    p = path.lower()
    if "dockerfile" in p:
        return "docker"
    return TEXT_LANG_BY_EXT.get(Path(p).suffix.lstrip("."), "text")

def _looks_binary(data: bytes) -> bool:
    # Heuristic: contains NUL or very high non-text ratio
    if b"\x00" in data:
        return True
    # treat small files as text
    if len(data) < 4:
        return False
    # rough ratio of non-printables
    sample = data[:4096]
    nonprint = sum(1 for b in sample if b < 9 or (13 < b < 32) or b == 127)
    return (nonprint / max(1, len(sample))) > 0.30

def _read_zip(file) -> List[Tuple[str, str]]:
    """Return list of (relpath, text) from a zip; includes files with no extension too; skips binary."""
    out: List[Tuple[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(file.getvalue()), "r") as zf:
            # basic guard (avoid huge zips with thousands of files on small instances)
            for i, info in enumerate(zf.infolist()):
                if i > 2000:
                    st.warning("ZIP contains more than 2000 files; extra files were skipped.")
                    break
                if info.is_dir():
                    continue
                raw = zf.read(info.filename)
                if _looks_binary(raw):
                    continue
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
                out.append((info.filename, text))
    except zipfile.BadZipFile:
        st.error(f"Invalid ZIP file: {getattr(file, 'name', 'uploaded.zip')}")
    return out

def _read_uploads(files) -> List[Tuple[str, str]]:
    """Return a flat list of (name, text) from uploaded files and zips; skips binary."""
    sources: List[Tuple[str, str]] = []
    for f in files or []:
        name = f.name or "uploaded"
        if name.lower().endswith(".zip"):
            sources.extend(_read_zip(f))
            continue
        raw = f.getvalue()
        if _looks_binary(raw):
            # silently skip binaries (images, pdfs, etc.)
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
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

    all_files: List[Dict[str, str]] = []
    all_notes: List[str] = []
    usage_merged: Dict[str, str] = {}
    missing_refs: List[str] = []

    with st.status("Converting…", expanded=False) as s:
        for idx, (fname, content) in enumerate(sources, start=1):
            s.update(label=f"Parsing {fname} ({idx}/{len(sources)})")
            structured = parse_code(content, file_name=fname)

            # detect legacy type (works for .txt or no extension)
            ltype = detect_legacy_type(structured)

            # basic missing reference heuristics
            up = structured.upper()
            if ltype == "cobol" and "COPY " in up:
                missing_refs.append(f"{fname}: contains COPY statements — ensure copybooks are uploaded.")
            if ltype == "jcl" and " DD " in f" {up} " and "DSN=" in up:
                missing_refs.append(f"{fname}: references datasets — verify control cards/inputs are present.")

            s.update(label=f"Calling model for {fname}…")
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
                    extra_context={"source_filename": fname, "detected_type": ltype},
                )
            except Exception as e:
                st.error(f"Conversion failed for {fname}: {e}")
                continue

            all_files.extend(bundle.get("files", []))
            if bundle.get("notes_markdown"):
                all_notes.append(f"### Notes for *{fname}*\n" + bundle["notes_markdown"])
            usage_merged = {**usage_merged, **bundle.get("usage", {})}

    st.success("✅ Conversion complete")

    # ---------------- Tabs ----------------
    tab_files, tab_diff, tab_notes, tab_usage, tab_chat = st.tabs(
        ["📦 Files", "🆚 Comparison", "📝 Notes", "📊 Usage", "💬 Chatbot"]
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

            # Build ZIP in-memory
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in all_files:
                    zf.writestr(f.get("path", "output.txt"), f.get("content", ""))
                if all_notes:
                    zf.writestr("MIGRATION_NOTES.md", "\n\n---\n\n".join(all_notes))
                if missing_refs:
                    zf.writestr("MISSING_REFERENCES.md", "\n".join(f"- {m}" for m in missing_refs))
            mem.seek(0)
            st.download_button(
                "⬇️ Download ZIP",
                data=mem,
                file_name="modernization_bundle.zip",
                mime="application/zip",
            )

    # ---------- Comparison ----------
    with tab_diff:
        st.write("#### Legacy vs Modernized (choose a modern file)")
        if not all_files:
            st.info("Generate a bundle to see comparison.")
        else:
            modern_paths = [f["path"] for f in all_files]
            chosen = st.selectbox("Modernized file", modern_paths)
            modern_content = next((f["content"] for f in all_files if f["path"] == chosen), "")

            # try to pick a matching legacy source by basename; fallback to first file
            base = Path(chosen).stem.lower()
            legacy_pick = ""
            for name, text in sources:
                if Path(name).stem.lower() == base:
                    legacy_pick = text
                    break
            if not legacy_pick and sources:
                legacy_pick = sources[0][1]

            diff_html = difflib.HtmlDiff().make_table(
                legacy_pick.splitlines(), modern_content.splitlines(),
                fromdesc="Legacy", todesc=chosen, context=True, numlines=5
            )
            st.components.v1.html(diff_html, height=600, scrolling=True)

    # ---------- Notes ----------
    with tab_notes:
        if all_notes:
            st.markdown("\n\n---\n\n".join(all_notes))
        else:
            st.info("No notes were returned. Enable *Migration Notes* in the sidebar.")
        if missing_refs:
            st.warning("*Missing/Dependent Inputs Detected:*\n\n" + "\n".join(f"- {m}" for m in missing_refs))

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
