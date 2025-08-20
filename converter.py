# converter.py
import os
import re
import json
from typing import Dict, List, Tuple, Optional

# Optional LLM SDKs (only used if provider selected and SDK installed)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from groq import Groq
except Exception:
    Groq = None

# ---------- Configuration ----------
DEFAULT_MAX_CHARS = 12000       # approx chunk size
CHUNK_OVERLAP = 800
DEFAULT_MAX_TOKENS = 4000

# ---------- Detection ----------
def detect_file_type(content: str) -> str:
    u = content.upper()
    # JCL detection
    if re.search(r"(?m)^\s*//\w+", u) or " JOB " in u:
        return "JCL"
    # COBOL detection
    if re.search(r"\bIDENTIFICATION\s+DIVISION\b|\bPROCEDURE\s+DIVISION\b|\bWORKING-STORAGE\b", u):
        return "COBOL"
    # CICS (COBOL with CICS)
    if "EXEC CICS" in u:
        return "COBOL_CICS"
    # DB2 embedded SQL
    if "EXEC SQL" in u:
        return "COBOL_DB2"
    # COPYBOOK heuristics
    if "DIVISION" not in u and re.search(r"(?m)^\s*01\s+\w+", u):
        return "COPYBOOK"
    # Control card (heuristic)
    if re.search(r"(?m)^\s*\w+\s*=\s*[^=]+$", content) or "PARM=" in u or "SORT FIELDS" in u:
        return "CONTROL_CARD"
    return "UNKNOWN"

# ---------- Chunking ----------
def chunk_text(content: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(content) <= max_chars:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        end = min(len(content), start + max_chars)
        chunk = content[start:end]
        chunks.append(chunk)
        if end == len(content):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- Providers wrappers ----------
def _call_openai(messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("MY_NEW_APP_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found — set MY_NEW_APP_KEY")
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages,
                                         temperature=float(temperature), max_tokens=int(max_tokens))
    return resp.choices[0].message.content

def _call_groq(messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq API key not found — set GROQ_API_KEY")
    if Groq is None:
        raise RuntimeError("groq SDK not installed")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages,
                                         temperature=float(temperature), max_tokens=int(max_tokens))
    return resp.choices[0].message.content

def llm_call(provider: str, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
    p = (provider or "openai").lower()
    if p == "openai":
        return _call_openai(messages, model, temperature, max_tokens)
    if p == "groq":
        return _call_groq(messages, model, temperature, max_tokens)
    raise RuntimeError("Unknown provider or provider not enabled")

# ---------- Simple stubs (if no LLM) ----------
def stub_cobol_to_java(content: str) -> str:
    return ("package com.example;\n\npublic class CobolConverted {\n"
            "    public static void main(String[] args) {\n"
            "        // Converted (stub) — replace with LLM output\n"
            "    }\n}\n")

def stub_jcl_to_groovy(content: str) -> str:
    return ("pipeline {\n  agent any\n  stages {\n    stage('job') { steps { echo 'stub' } }\n  }\n}\n")

def stub_jcl_to_shell(content: str) -> str:
    return ("#!/usr/bin/env bash\nset -euo pipefail\n# stub converted JCL\n")

def stub_copybook_to_pojo(content: str) -> str:
    return ("public class CopybookDTO {\n    // fields to be generated from copybook\n}\n")

def stub_control_to_conf(content: str) -> str:
    return ("# Control card converted stub\n" + content + ("\n" if not content.endswith("\n") else ""))

# ---------- Helpers ----------
def find_copybook_refs(content: str) -> List[str]:
    refs = re.findall(r"(?i)\bCOPY\s+['\"]?([A-Z0-9_#\$@\-\.]+)['\"]?", content)
    return [r.upper() for r in refs]

def format_by_extension(path: str, text: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    ext = ext.lower()
    # Very lightweight formatters — they help avoid single-line blobs:
    if ext in ("java", "groovy", "js", "cs"):
        # Indent based on braces
        out_lines = []
        indent = 0
        for raw in text.splitlines():
            line = raw.rstrip()
            if line.strip().startswith("}"):
                indent = max(indent - 1, 0)
            out_lines.append(("    " * indent) + line)
            indent += line.count("{") - line.count("}")
        res = "\n".join(out_lines).rstrip() + "\n"
        return res
    # For python, yaml, conf, keep lines but strip trailing spaces
    return "\n".join(l.rstrip() for l in text.splitlines()) + ("\n" if text and not text.endswith("\n") else "")

# ---------- Core conversion for a single file ----------
def _build_system_message_for_type(ftype: str) -> str:
    # Short system guidance for the LLM; not strict JSON mode — request plain text outputs
    if ftype in ("COBOL", "COBOL_CICS", "COBOL_DB2"):
        return ("You are a senior engineer converting COBOL programs to modern languages. "
                "Produce only well-formatted code (no markdown fences), include imports/package skeletons, "
                "and add comments where assumptions are made.")
    if ftype == "JCL":
        return ("You are a JCL -> Shell/Groovy conversion expert. Map steps to pipeline stages or bash functions. "
                "Produce only the pipeline or shell code (no markdown).")
    if ftype == "COPYBOOK":
        return ("You are a copybook converter: produce a strongly-typed Java POJO or Python Pydantic model. "
                "Return code only, properly formatted.")
    if ftype == "CONTROL_CARD":
        return ("Convert this control card to a .conf/.properties style file or clear comments + mapping.")
    return ("Transform the input into a well-formatted modern artifact as requested. Return code/text only.")

def convert_chunk_with_llm(provider: str, model: str, system_msg: str, user_content: str,
                           temperature: float, max_tokens: int) -> str:
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}]
    return llm_call(provider, messages, model, temperature, max_tokens)

def convert_single_file(
    file_path: str,
    file_content: str,
    ftype: str,
    target_hint: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_chars: int,
) -> Tuple[List[Dict[str,str]], List[str]]:
    """
    Returns (list_of_files_dicts, missing_copybooks)
    Each file dict: {"path": "relative/path.ext", "content": "text"}
    """
    # detect copybook refs if COBOL
    missing = []
    if ftype.startswith("COBOL"):
        refs = find_copybook_refs(file_content)
        # caller will provide known list to compare; here we just return refs as placeholders
        missing = refs  # the caller may filter which are missing

    # choose default output path extension
    if ftype.startswith("COBOL"):
        out_ext = "java"
    elif ftype == "JCL":
        out_ext = "groovy" if target_hint == "groovy" else "sh"
    elif ftype == "COPYBOOK":
        out_ext = "java"
    elif ftype == "CONTROL_CARD":
        out_ext = "conf"
    else:
        out_ext = "txt"

    chunks = chunk_text(file_content, max_chars=max_chars, overlap=CHUNK_OVERLAP)
    results = []
    for idx, ch in enumerate(chunks, 1):
        # build user content with instructions and chunk info
        user_content = (
            f"Target: {target_hint}\n"
            f"Chunk: {idx}/{len(chunks)}\n"
            "Input:\n"
            + ch
            + ("\n\nNote: This output should be code only, formatted, without markdown fences.")
        )

        # select system message
        sys_msg = _build_system_message_for_type(ftype)
        if provider and provider.lower() != "none":
            try:
                out_text = convert_chunk_with_llm(provider, model, sys_msg, user_content, temperature, max_tokens)
            except Exception as e:
                # if LLM fails, fallback to stub
                out_text = f"// LLM call failed: {e}\n" + (stub_cobol_to_java(ch) if ftype.startswith("COBOL") else ch)
        else:
            # non-LLM fallback
            if ftype.startswith("COBOL"):
                out_text = stub_cobol_to_java(ch)
            elif ftype == "JCL":
                out_text = stub_jcl_to_groovy(ch) if target_hint == "groovy" else stub_jcl_to_shell(ch)
            elif ftype == "COPYBOOK":
                out_text = stub_copybook_to_pojo(ch)
            elif ftype == "CONTROL_CARD":
                out_text = stub_control_to_conf(ch)
            else:
                out_text = ch

        # path hint (the caller will map to real output path)
        path_hint = f"converted_part_{idx}.{out_ext}"
        formatted = format_by_extension(path_hint, out_text)
        results.append({"path": path_hint, "content": formatted})

    # merge parts with simple concatenation by default (caller can rename/organize)
    merged_by_path: Dict[str, str] = {}
    for r in results:
        p = r["path"]
        merged_by_path.setdefault(p, "")
        merged_by_path[p] += r["content"] + ("\n" if not r["content"].endswith("\n") else "")

    files = [{"path": p, "content": c} for p, c in merged_by_path.items()]
    return files, missing

# ---------- High-level project conversion ----------
def convert_project(
    files: Dict[str, str],
    target_stack: str,
    instructions: str,
    requested_artifacts: List[str],
    provider: str = "none",
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Dict:
    """
    files: dict of relative_path -> content
    returns bundle dict:
    {
      "files": [{"path": outpath, "content": "..."}],
      "notes_markdown": "...",
      "usage": {...},
      "missing": { "copybooks": [...] }
    }
    """
    out_files = []
    notes = []
    missing_copybooks = set()

    # pre-scan known copybooks present
    present_copybooks = set()
    for p, c in files.items():
        if detect_file_type(c) == "COPYBOOK":
            # attempt to pick top-level 01 name
            m = re.search(r"(?m)^\s*01\s+([A-Z0-9_]+)", c, re.IGNORECASE)
            if m:
                present_copybooks.add(m.group(1).upper())

    # process each file
    for relpath, content in files.items():
        ftype = detect_file_type(content)
        # determine per-file target hint (JCL: groovy or shell; COBOL: target_stack)
        if ftype == "JCL":
            jcl_hint = "groovy" if "groovy" in (target_stack or "").lower() else "shell"
            target_hint = jcl_hint
        else:
            target_hint = target_stack or "java"

        converted_files, missing = convert_single_file(
            file_path=relpath,
            file_content=content,
            ftype=ftype,
            target_hint=target_hint,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )
        # map converted file paths to logical output paths
        for idx, f in enumerate(converted_files):
            # produce clear output paths preserving source base name
            base = os.path.splitext(os.path.basename(relpath))[0]
            # if converted piece path contains an index, we keep it
            ext = f["path"].rsplit(".", 1)[-1] if "." in f["path"] else "txt"
            if ftype.startswith("COBOL"):
                outpath = f"outputs/cobol_java/{base}.{ext}"
            elif ftype == "JCL":
                if target_hint == "groovy":
                    outpath = f"outputs/jcl_groovy/{base}.{ext}"
                else:
                    outpath = f"outputs/jcl_shell/{base}.{ext}"
            elif ftype == "COPYBOOK":
                outpath = f"outputs/copybook_java/{base}.{ext}"
            elif ftype == "CONTROL_CARD":
                outpath = f"outputs/control_cards/{base}.{ext}"
            else:
                outpath = f"outputs/other/{base}.{ext}"
            out_files.append({"path": outpath, "content": f["content"]})

        # collect missing only if referenced but not present
        for ref in missing:
            if ref.upper() not in present_copybooks:
                missing_copybooks.add(ref)

    notes_text = ""
    if missing_copybooks:
        notes_text += "### Missing copybooks detected\n" + "\n".join(f"- {c}" for c in sorted(missing_copybooks)) + "\n"

    usage = {"provider": provider, "model": model, "files_in": len(files), "files_out": len(out_files)}
    bundle = {"files": out_files, "notes_markdown": notes_text or "No migration notes.", "usage": usage, "missing": {"copybooks": sorted(missing_copybooks)}}
    return bundle
