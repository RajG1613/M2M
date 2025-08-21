# converter.py
import os
import re
import json
from typing import Dict, List, Any, Iterable

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Groq SDK
try:
    from groq import Groq
except Exception:
    Groq = None


SYSTEM_PROMPT = """
You are an expert in Mainframe Modernization.

Special Rules:
- If legacy type is JCL:
   * Default => convert to Shell scripts (bash) with correct Identation code that mirrors steps/DDs.
   * If target stack contains 'groovy' => convert to Groovy.
- COBOL => convert into the requested target stack (Spring/Java, FastAPI/Python, .NET, Node).
- DB2 => modern SQL/ORM.
- VSAM => relational/NoSQL schema + data access code.
- CICS => REST APIs (controller + service layer) in the chosen stack (document transaction mappings).
- IMS DB => RDBMS schema + migration utilities.

Formatting Rules (MUST FOLLOW):
- Output well-formatted, readable files (proper line breaks and indentation).
- NEVER return code as a single line.
- Organize artifacts using conventional project structure for the chosen stack.
- Unit tests go into a conventional test folder.
- Add concise comments where business logic is inferred.
- For each file, return: {"path": "...", "content": "..."}.

Respond with the converted code ONLY, in the exact format of the target language. 
Do not include explanations, markdown, or JSON. 
Output should be clean and ready to save as a file. 
If multiple files are required, provide each one separately.
"""


# ----------------- public small util for UI -----------------
def detect_legacy_type(legacy_code: str) -> str:
    """Lightweight detector used by UI too (works for .txt or no extension)."""
    code_upper = legacy_code.upper()
    if code_upper.strip().startswith("//") or " JOB " in code_upper or "//STEP" in code_upper:
        return "jcl"
    if "PROCEDURE DIVISION" in code_upper or "IDENTIFICATION DIVISION" in code_upper:
        return "cobol"
    if "COPY " in code_upper and "REPLACING" in code_upper:
        # still COBOL, copybook usage
        return "cobol"
    if "CICS" in code_upper or "EXEC CICS" in code_upper:
        return "cics"
    if "EXEC SQL" in code_upper:
        return "db2"
    if "IMS" in code_upper and "PCB" in code_upper:
        return "ims"
    return "unknown"


def _stack_defaults(target_stack: str) -> Dict[str, str]:
    ts = target_stack.lower()
    if "spring" in ts:  # Java/Spring Boot
        return {"main": "src/main/java/com/example/App.java", "test_dir": "src/test/java/com/example"}
    if "fastapi" in ts:  # Python
        return {"main": "app/main.py", "test_dir": "tests"}
    if ".net" in ts or "c#" in ts:
        return {"main": "src/Program.cs", "test_dir": "tests"}
    if "node" in ts or "express" in ts:
        return {"main": "src/index.js", "test_dir": "_tests_"}
    if "groovy" in ts:  # JCL -> Groovy
        return {"main": "File.groovy", "test_dir": "tests"}
    if "shell" in ts:  # JCL -> shell
        return {"main": "scripts/job.sh", "test_dir": "tests"}
    return {"main": "src/output.txt", "test_dir": "tests"}


# ----------------- LLM calls -----------------
def _call_openai(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("MY_NEW_APP_KEY")
    if not api_key:
        raise RuntimeError("OpenAI key not found: set env MY_NEW_APP_KEY")
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Add openai to requirements.txt")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content


def _call_groq(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq key not found: set env GROQ_API_KEY")
    if Groq is None:
        raise RuntimeError("groq SDK not installed. Add groq to requirements.txt")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content


# ----------------- chunking & merge -----------------
def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 600) -> Iterable[str]:
    """Split very large sources into overlapping chunks (by chars)."""
    if len(text) <= max_chars:
        yield text
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        yield text[start:end]
        if end == n:
            break
        start = end - overlap


def _merge_bundles(bundles: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"files": [], "notes_markdown": "", "usage": {}}
    seen_paths = set()
    notes: List[str] = []
    for b in bundles:
        for f in b.get("files", []):
            path = f.get("path") or "output.txt"
            # disambiguate duplicates by chunk: keep first and append others with suffix
            if path in seen_paths:
                base, ext = (path.rsplit(".", 1) + [""])[:2]
                path = f"{base}_part{len(seen_paths)}" + (("." + ext) if ext else "")
            seen_paths.add(path)
            merged["files"].append({"path": path, "content": f.get("content", "")})
        nm = b.get("notes_markdown")
        if nm:
            notes.append(nm)
        merged["usage"] = {**merged["usage"], **b.get("usage", {})}
    merged["notes_markdown"] = "\n\n---\n\n".join(notes)
    return merged


# ----------------- formatting guardrails -----------------
def _is_single_liney(s: str) -> bool:
    # treat minified/line-collapsed content
    return ("\n" not in s) or (len(max(s.splitlines(), key=len, default="")) > 300)

def _pretty_format(path: str, content: str) -> str:
    """Non-destructive heuristics to avoid single-line blobs and improve readability."""
    if not _is_single_liney(content):
        return content

    # Generic fallbacks
    c = content

    # add newlines after common statement terminators/blocks
    c = re.sub(r";(?!\s*\n)", ";\n", c)
    c = re.sub(r"\{", "{\n", c)
    c = re.sub(r"\}(?!\s*\n)", "\n}\n", c)
    c = re.sub(r"\)\s*\{", "){\n", c)

    # language specific hints
    lower = path.lower()
    if lower.endswith(".py"):
        c = c.replace("def ", "\n\ndef ")
        c = c.replace("class ", "\n\nclass ")
    if lower.endswith((".java", ".cs", ".js", ".groovy")):
        c = re.sub(r"(public|private|protected|class|interface|return|if|else|for|while)\s", r"\n\1 ", c)
    if lower.endswith(".sh"):
        c = c.replace(" && ", " && \\\n")
    if lower.endswith((".yml", ".yaml")):
        c = re.sub(r"(\w+:)", r"\n\1", c)

    # ensure not one giant line
    lines = c.splitlines()
    out = []
    indent = 0
    for line in lines:
        line = line.rstrip()
        # crude brace-based indentation
        if re.search(r"^\s*[\}\)]", line):
            indent = max(0, indent - 1)
        out.append(("    " * indent) + line.lstrip())
        if re.search(r"\{\s*$", line):
            indent += 1
    return "\n".join(out).strip() + "\n"


# ----------------- main API -----------------
def convert_to_bundle(
    legacy_code: str,
    target_stack: str,
    instructions: str,
    requested_artifacts: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str = "OpenAI",
    extra_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Convert legacy code into modern artifacts.
    Handles large inputs via chunking, and post-formats outputs to avoid single-line blobs.
    """
    legacy_type = detect_legacy_type(legacy_code)

    # Override JCL conversion defaults
    if legacy_type == "jcl":
        if "groovy" in target_stack.lower():
            target_stack = "Groovy"
        elif "shell" in target_stack.lower():
            target_stack = "Shell Script"
        else:
            target_stack = "Shell Script"

    defaults = _stack_defaults(target_stack)

    # chunk if needed
    bundles: List[Dict[str, Any]] = []
    chunks = list(_chunk_text(legacy_code))
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        user_payload = {
            "target_stack": target_stack,
            "instructions": instructions,
            "requested_artifacts": requested_artifacts,
            "parsed_legacy": chunk,
            "legacy_type": legacy_type,
            "defaults": defaults,
            "chunk": {"index": i, "total": total},
        }
        if extra_context:
            user_payload["context"] = extra_context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

        content = _call_groq(messages, model, temperature, max_tokens) if provider.lower() == "groq" \
            else _call_openai(messages, model, temperature, max_tokens)

        try:
            bundle = json.loads(content)
            if "files" not in bundle:
                raise ValueError("No 'files' in JSON")
        except Exception:
            bundle = {
                "files": [{"path": defaults["main"], "content": content}],
                "notes_markdown": f"⚠️ Model did not return JSON. Wrapped raw output for chunk {i}/{total}.",
                "usage": {}
            }

        # post-format each file
        for f in bundle.get("files", []):
            p = f.get("path", "output.txt")
            f["content"] = _pretty_format(p, f.get("content", ""))

        # mark which chunk created the file (useful if same path repeats)
        for f in bundle.get("files", []):
            if total > 1 and f.get("path"):
                # annotate note, not path (keep path clean); note will be merged
                note_line = f"- Chunk {i}/{total} produced {f['path']}."
                nm = bundle.get("notes_markdown", "")
                bundle["notes_markdown"] = (nm + ("\n" if nm else "") + note_line).strip()

        # usage meta
        usage = bundle.get("usage", {})
        usage.update({"provider": provider, "model": model})
        bundle["usage"] = usage

        bundles.append(bundle)

    # merge chunks
    final_bundle = _merge_bundles(bundles)

    # final pass: ensure formatting
    for f in final_bundle.get("files", []):
        f["content"] = _pretty_format(f.get("path", "output.txt"), f.get("content", ""))

    # add usage meta
    usage = final_bundle.get("usage", {})
    usage.update({"provider": provider, "model": model})
    final_bundle["usage"] = usage
    return final_bundle


# -------- Interactive Chatbot --------
def chatbot(
    legacy_code: str,
    query: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    provider: str = "Groq"
) -> str:
    """
    Interactive chatbot for modernization Q&A.
    Example: explain COBOL logic, map DB2/VSAM/IMS to modern stores, JCL step mapping, etc.
    """
    legacy_type = detect_legacy_type(legacy_code)
    messages = [
        {"role": "system", "content": f"You are a modernization assistant. Legacy type = {legacy_type}."},
        {"role": "user", "content": f"Legacy code (combined):\n{legacy_code}\n\nUser question:\n{query}\n\nPlease answer clearly and concisely."}
    ]

    if provider.lower() == "groq":
        return _call_groq(messages, model, temperature, max_tokens)
    else:
        return _call_openai(messages, model, temperature, max_tokens)

