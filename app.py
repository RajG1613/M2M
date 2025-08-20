# app.py
import os
import io
import zipfile
import tempfile
import shutil
import streamlit as st
import difflib
from typing import Dict
from converter import convert_project, detect_file_type

st.set_page_config(page_title="AI Mainframe Modernizer", layout="wide")

st.title("🚀 AI Mainframe Modernizer")
st.write("Upload a zip/folder of legacy sources (COBOL, JCL, Copybooks). The tool auto-detects and modernizes.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["none", "OpenAI", "Groq"])
    model = st.text_input("Model", value="gpt-4o")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=512, max_value=16000, value=4000, step=256)
    max_chars = st.number_input("Chunk max chars", min_value=2000, max_value=50000, value=12000, step=1000)
    st.divider()
    st.caption("Upload a .zip containing project files. Files are detected by content, not extension.")

uploaded = st.file_uploader("Upload .zip file of legacy sources", type=["zip"], accept_multiple_files=False)

if uploaded:
    temp_dir = tempfile.mkdtemp(prefix="legacy_")
    try:
        zip_bytes = uploaded.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(temp_dir)

        # read all small files into dict
        files: Dict[str, str] = {}
        for root, _, fns in os.walk(temp_dir):
            for fn in fns:
                path = os.path.join(root, fn)
                # skip binary files > 10MB to avoid memory issues; they can be added later with streaming
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                except Exception:
                    content = ""
                rel = os.path.relpath(path, temp_dir)
                files[rel] = content

        st.success(f"Unzipped {len(files)} files.")

        if st.button("Start Modernization"):
            with st.spinner("Converting..."):
                bundle = convert_project(
                    files=files,
                    target_stack="Java (Spring Boot)",
                    instructions="Produce production-grade artifacts. Preserve business logic.",
                    requested_artifacts=["Unit Tests", "OpenAPI Spec"],
                    provider=(provider or "none").lower(),
                    model=model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    max_chars=int(max_chars),
                )

            st.success("Conversion complete!")

            # Show missing copybooks if any
            missing = bundle.get("missing", {}).get("copybooks", [])
            if missing:
                st.warning("Missing copybooks:\n" + ", ".join(missing))

            files_out = bundle.get("files", [])
            notes = bundle.get("notes_markdown", "")
            usage = bundle.get("usage", {})

            tab_files, tab_compare, tab_notes, tab_chat = st.tabs(["Files", "Compare (side-by-side)", "Notes", "Chatbot"])

            # Files tab: list and show
            with tab_files:
                st.write("### Generated files")
                if not files_out:
                    st.info("No files generated.")
                else:
                    for f in files_out:
                        st.write(f"*{f['path']}*")
                        # language guess
                        ext = f['path'].rsplit(".", 1)[-1] if "." in f['path'] else "txt"
                        lang = "text"
                        if ext in ("java",): lang = "java"
                        if ext in ("py",): lang = "python"
                        if ext in ("groovy",): lang = "groovy"
                        if ext in ("yml","yaml"): lang = "yaml"
                        if ext in ("sh",): lang = "bash"
                        st.code(f['content'], language=lang)
                    # allow download as zip
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                        for f in files_out:
                            zf.writestr(f['path'], f['content'])
                        if notes:
                            zf.writestr("MIGRATION_NOTES.md", notes)
                    mem.seek(0)
                    st.download_button("Download all outputs (zip)", mem, file_name="modernization_outputs.zip")

            # Side-by-side compare: allow user pick a legacy file and a generated file
            with tab_compare:
                st.write("### Side-by-side comparison (legacy vs modernized)")
                legacy_choice = st.selectbox("Legacy file", options=list(files.keys()))
                created_paths = [f['path'] for f in files_out]
                created_choice = st.selectbox("Converted file", options=(created_paths or ["No outputs yet"]))
                left = files.get(legacy_choice, "")
                right = ""
                for f in files_out:
                    if f['path'] == created_choice:
                        right = f['content']
                        break
                if st.button("Show diff"):
                    if not right:
                        st.error("Selected converted file empty.")
                    else:
                        # build HTML diff with preserved indentation
                        diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
                            left.splitlines(), right.splitlines(),
                            fromdesc="Legacy", todesc="Modernized", context=True, numlines=5
                        )
                        st.components.v1.html(diff_html, height=600, scrolling=True)

            # Notes tab
            with tab_notes:
                st.markdown("### Migration Notes")
                st.markdown(notes or "No notes returned.")

            # Chatbot tab — Q&A about combined legacy code
            with tab_chat:
                st.write("### Chatbot: ask about the uploaded legacy code or conversion")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                for role, msg in st.session_state.chat_history:
                    if role == "user":
                        st.markdown(f"*You:* {msg}")
                    else:
                        st.markdown(f"*Assistant:* {msg}")

                ask = st.text_input("Ask a question about the uploaded project:")
                if st.button("Send") and ask.strip():
                    st.session_state.chat_history.append(("user", ask))
                    # use simple LLM chat via convert_project (we'll call llm directly using conversion helper)
                    try:
                        # We pack combined content as single piece for chatbot context
                        combined = "\n\n".join((k + ":\n" + v[:2000]) for k, v in list(files.items())[:20])
                        # Build a short messages payload and call LLM via convert_project's llm if provider set
                        from converter import detect_file_type  # already imported at module top
                        system = "You are a modernization assistant. Answer concisely and clearly."
                        user_content = f"Context (first 20 files truncated):\n{combined}\n\nUser question:\n{ask}"
                        if provider.lower() in ("openai", "groq"):
                            # call LLM directly using same interface as converter
                            import converter as cv
                            out = cv.llm_call(provider.lower(),
                                              [{"role":"system","content":system},{"role":"user","content":user_content}],
                                              model, float(temperature), int(max_tokens))
                        else:
                            out = "(Chatbot disabled — set provider to OpenAI or Groq for live answers.)"
                    except Exception as e:
                        out = f"(Chatbot error: {e})"
                    st.session_state.chat_history.append(("assistant", out))
                    st.experimental_rerun()

    finally:
        # cleanup temp dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
