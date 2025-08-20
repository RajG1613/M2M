# app.py
import io
import os
import re
import time
import difflib
import zipfile
import textwrap
import json
import xml.dom.minidom as minidom
from typing import List, Dict, Tuple

import streamlit as st
from converter import convert_to_bundle, chatbot
from parser import parse_code

# ---------- Streamlit page ----------
st.set_page_config(page_title="AI Mainframe Modernizer", page_icon="🚀", layout="wide")

st.title("🚀 AI Mainframe Modernizer — Enterprise Demo")
st.caption("Upload COBOL/JCL (single, multiple, or ZIP) → get modern artifacts: code, tests, API spec, CI pipeline, Dockerfile & migration notes.")

# ---------- Helpers ----------
def _read_uploads(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict[str, str]:
    """
    Returns {relative_path: text_content}. Supports plain files and a single .zip.
    """
    result: Dict[str, str] = {}
    if not files:
        return result

    # If user sent a single ZIP, expand it
    if len(files) == 1 and files[0].name.lower().endswith(".zip"):
        data = files[0].getvalue()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for zinfo in zf.infolist():
                if zinfo.is_dir():
                    continue
                name = zinfo.filename
                # Only read smallish text-like files; skip binaries
                try:
                    text = zf.read(zinfo).decode("utf-8", errors="replace")
                except Exception:
                    continue
                result[name] = text
        return result

    # Otherwise treat as multiple individual files
    for f in files:
        try:
            result[f.name] = f.getvalue().decode("utf-8", errors="replace")
        except Exception:
            # ignore non-decodable files
            pass
    return result


def _combined_legacy(files_map: Dict[str, str]) -> str:
    """
    Build a single text block that keeps per-file boundaries for model context & diff.
    """
    parts = []
    for path, txt in files_map.items():
        parts.append(f"\n===== BEGIN FILE: {path} =====\n{txt}\n===== END FILE: {path} =====\n")
    return "\n".join(parts).strip()


def _detect_missing_refs(files_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Very lightweight detection of likely missing include/control card references:
    - COBOL: COPY <name>
    - JCL: //SYSIN DD *, INCLUDE, or //DD statements referencing external DSNs
    Returns: {"copybooks": [...], "includes": [...], "ddnames": [...]} (values possibly empty)
    """
    uploaded_names = set([os.path.splitext(os.path.basename(p))[0].upper() for p in files_map.keys()])

    copybooks, includes, ddnames = set(), set(), set()

    cobol_copy_re = re.compile(r"\bCOPY\s+([A-Z0-9_#\-]+)", re.IGNORECASE)
    jcl_include_re = re.compile(r"\bINCLUDE\s+([A-Z0-9_#\-/\.]+)", re.IGNORECASE)
    jcl_dd_re = re.compile(r"^\s*//([A-Z0-9$#@]+)\s+DD\b.*", re.IGNORECASE | re.MULTILINE)

    for path, txt in files_map.items():
        up = txt.upper()

        # COBOL COPY
        for m in cobol_copy_re.findall(up):
            copybooks.add(m.upper())

        # JCL INCLUDE (often used to pull control statements)
        for m in jcl_include_re.findall(up):
            includes.add(m.upper())

        # JCL DD names – we just list them; resolving DSN/HLQ is site specific
        for m in jcl_dd_re.findall(up):
            ddnames.add(m.upper())

    # Treat a COPY <X> as "present" if any uploaded file’s stem name matches X
    missing_copybooks = [c for c in sorted(copybooks) if c not in uploaded_names]
    # Includes are often members; we cannot verify well, but still report
    missing_includes = sorted(includes)

    return {
        "copybooks": missing_copybooks,
        "includes": missing_includes,
        "ddnames": sorted(ddnames),
    }


def _guess_lang(path: str) -> str:
    p = path.lower()
    if "dockerfile" in p:
        return "docker"
    ext = os.path.splitext(p)[1].lstrip(".")
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
        "bat": "bat",
    }.get(ext, "text")


def _format_content_auto(path: str, content: str) -> str:
    """
    Safe, dependency-free pretty printing & de-minifying so you don’t get one-line blobs.
    Not perfect like real formatters, but much better for demos.
    """
    lang = _guess_lang(path)
    text = content or ""

    # Quick fixes first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # JSON
    if lang == "json":
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            return text

    # XML
    if lang == "xml":
        try:
            dom = minidom.parseString(text.encode("utf-8"))
            return dom.toprettyxml(indent="  ")
        except Exception:
            return text

    # YAML: avoid adding PyYAML dependency; lightly indent blocks
    if lang == "yaml":
        # heuristic: ensure there’s a newline after colons when needed
        fixed = re.sub(r":(\S)", r": \1", text)
        return fixed

    # Bash/Groovy/Java/C#/JS: basic brace indentation
    if lang in {"groovy", "java", "csharp", "javascript"}:
        indent = 0
        out_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.endswith("}") or stripped.startswith("}"):
                indent = max(0, indent - 1)
            out_lines.append(("  " * indent) + stripped)
            if stripped.endswith("{") and not stripped.endswith("{}"):
                indent += 1
        return "\n".join(out_lines)

    # Python: add newlines around def/class and indent blocks heuristically
    if lang == "python":
        # add newline after colon that starts a block if missing
        return "\n".join(textwrap.dedent(text).splitlines())

    # Default: collapse multiple blank lines and trim
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"
    return text


def _limit_hint():
    st.info(
        "📦 To allow up to **1GB** uploads, create `.streamlit/config.toml` in your repo with:\n\n"
        "```\n[server]\nmaxUploadSize = 1000\n```\n"
        "On some hosts you may also need to raise proxy limits."
    )

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Groq"])

    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        key_hint = "🔐 Reads OpenAI key from env var **MY_NEW_APP_KEY**"
        missing = not os.getenv("MY_NEW_APP_KEY")
    else:
        model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        key_hint = "🔐 Reads Groq key from env var **GROQ_API_KEY**"
        missing = not os.getenv("GROQ_API_KEY")

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max output tokens", min_value=512, max_value=8192, value=4096, step=256)

    st.divider()
    target_stack = st.selectbox(
        "Target stack",
        [
            "Java (Spring Boot)",
            "Python (FastAPI)",
            "C# (.NET)",
            "Node.js (Express)",
            "Groovy (for JCL)",   # JCL → Groovy pipelines
            "Shell (for JCL)",    # JCL → Shell scripts
            "REST APIs",
        ],
    )
    extras = st.multiselect(
        "Artifacts to include",
        ["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "K8s Manifests", "Migration Notes"],
        default=["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "Migration Notes"]
    )
    st.divider()
    _limit_hint()
    st.caption(key_hint)
    if missing:
        st.warning("⚠️ No API key detected for the selected provider.")

# ---------------- Upload + Prompt ----------------
st.write("### 1) Upload legacy files (one, many, or a ZIP folder)")
uploads = st.file_uploader(
    "Supported: .cbl, .cob, .jcl, .txt, .zip",
    type=["cbl", "cob", "jcl", "txt", "zip"],
    accept_multiple_files=True
)

st.write("### 2) Modernization instructions")
default_prompt = (
    "Convert to production-ready code using the selected stack. "
    "Preserve business logic, remove dead code, and use best practices. "
    "Map DB2/VSAM/IMS to modern data stores. Convert CICS to REST APIs. "
    "For JCL, convert to Shell or Groovy pipelines as appropriate. "
    "If data structures are implicit, make them explicit. Add comments where intent is unclear."
)
user_prompt = st.text_area("Prompt", value=default_prompt, height=140)

run = st.button("🛠️ Generate Modernization Bundle", type="primary", use_container_width=True)

# ---------------- Execution ----------------
if run:
    files_map = _read_uploads(uploads)
    if not files_map:
        st.error("Please upload at least one file (or a ZIP).")
        st.stop()
    if not user_prompt.strip():
        st.error("Please enter your instructions.")
        st.stop()

    combined_legacy = _combined_legacy(files_map)
    missing_refs = _detect_missing_refs(files_map)

    with st.status("Parsing legacy code…", expanded=False) as s:
        # If you have a smarter parser, you can feed each file; here we collapse for the prompt.
        structured = parse_code(combined_legacy, file_name="MULTI")
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
            provider=provider,
            extra_context={
                "uploaded_files": list(files_map.keys()),
                "missing_references": missing_refs,
            },
        )
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        st.stop()

    st.success("✅ Bundle ready!")

    files: List[Dict[str, str]] = bundle.get("files", [])
    notes = bundle.get("notes_markdown", "")
    usage = bundle.get("usage", {})

    # ---------------- Tabs ----------------
    tab_files, tab_diff, tab_missing, tab_notes, tab_usage, tab_chat = st.tabs(
        ["📦 Files", "🆚 Comparison", "❗ Missing", "📝 Notes", "📊 Usage", "💬 Chatbot"]
    )

    # ---------- Files ----------
    with tab_files:
        if not files:
            st.info("No files returned.")
        else:
            st.write("#### Generated files")
            formatted_files: List[Tuple[str, str]] = []
            for f in files:
                path = f.get("path", "output.txt")
                content = f.get("content", "")
                pretty = _format_content_auto(path, content)
                formatted_files.append((path, pretty))
                with st.expander(path, expanded=True):
                    st.code(pretty, language=_guess_lang(path))

            # ZIP download
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for path, pretty in formatted_files:
                    zf.writestr(path, pretty)
                if notes:
                    zf.writestr("MIGRATION_NOTES.md", notes)
            mem.seek(0)
            st.download_button("⬇️ Download ZIP", data=mem, file_name="modernization_bundle.zip", mime="application/zip")

    # ---------- Comparison ----------
    with tab_diff:
        st.write("#### Legacy vs Modernized")
        if files:
            modern_code = "\n".join(_format_content_auto(f.get("path", ""), f.get("content", "")) for f in files)
            diff = difflib.HtmlDiff().make_table(
                combined_legacy.splitlines(), modern_code.splitlines(),
                fromdesc="Legacy (combined)", todesc="Modernized (combined)", context=True, numlines=5
            )
            st.components.v1.html(diff, height=600, scrolling=True)
        else:
            st.info("Generate a bundle to see comparison.")

    # ---------- Missing ----------
    with tab_missing:
        st.write("#### Detected references")
        st.json(missing_refs)
        if missing_refs.get("copybooks") or missing_refs.get("includes"):
            st.warning(
                "The conversion can proceed, but results may be partial/approximate if copybooks or includes are missing."
            )

    # ---------- Notes ----------
    with tab_notes:
        if notes:
            st.markdown(notes)
        else:
            st.info("No notes were returned. Enable **Migration Notes** in the sidebar.")

    # ---------- Usage ----------
    with tab_usage:
        st.write("Token/provider usage reported by the API (if available).")
        st.json(usage or {"info": "No usage available"})

    # ---------- Chatbot ----------
    with tab_chat:
        st.write("#### Ask questions about your code/migration")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        if prompt := st.chat_input("Ask me anything about modernization…"):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    answer = chatbot(
                        legacy_code=combined_legacy,
                        query=prompt,
                        model=model,
                        temperature=float(temperature),
                        max_tokens=1200,
                        provider=provider,
                    )
                except Exception as e:
                    answer = f"Error: {e}"
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

