# converter.py
import os
import re
import json
from typing import Dict, List, Any, Iterable, Tuple

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


# ===========================
# System prompt (no strict JSON)
# ===========================
SYSTEM_PROMPT = """
You are an expert in Mainframe Modernization.

CONVERSION RULES
- JCL
  * Default: convert steps/DDs to portable Shell (bash) or CI/CD YAML mirroring the flow.
  * If target stack mentions 'groovy': produce Jenkins Groovy pipeline.
- COBOL: convert to the chosen stack (Spring/Java, FastAPI/Python, .NET, Node) with idiomatic structure.
- DB2 (EXEC SQL): modern SQL/ORM for the target stack.
- VSAM: relational/NoSQL schema + data access code for the stack.
- CICS (EXEC CICS): REST APIs + controller/service layers (document transaction mappings).
- IMS DB: RDBMS schema + migration utilities.

OUTPUT RULES (MUST FOLLOW)
- Return ONLY a series of <file> blocks plus an optional <notes> block.
- NO Markdown code fences, NO extra commentary, NO JSON.
- For each file:
    <file path="relative/path/with/extension">
    ...well-formatted code/content...
    </file>
- Optional notes:
    <notes>
    - bullet points for decisions/assumptions/risks
    </notes>

STYLE
- Files must be well-formatted, readable (correct indentation, line breaks).
- Use conventional project layout for the chosen stack (imports, package structure).
- Place unit tests in a conventional test folder for the stack.
- Add concise comments where business logic was inferred.
"""


# ===========================
# Utilities
# ===========================
def detect_legacy_type(legacy_code: str) -> str:
    """Detect legacy type purely from content (works for txt or no extension)."""
    code_upper = legacy_code.upper()

    # Quick JCL indicators
    if code_upper.strip().startswith("//") or " JOB " in code_upper or re.search(r"^\s*//STEP", code_upper, re.M):
        return "jcl"

    # COBOL divisions
    if "IDENTIFICATION DIVISION" in code_upper or "PROCEDURE DIVISION" in code_upper:
        return "cobol"

    # Copybook fragments (no PROCEDURE DIVISION)
    if re.search(r"^\s*01\s+[A-Z0-9-]+\.", code_upper, re.M) and "PROCEDURE DIVISION" not in code_upper:
        return "copybook"

    # CICS/DB2 keywords
    if "EXEC CICS" in code_upper:
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
        return {"main": "jenkins/Jenkinsfile.groovy", "test_dir": "tests"}
    if "shell" in ts or "bash" in ts:  # JCL -> shell
        return {"main": "scripts/job.sh", "test_dir": "tests"}
    return {"main": "src/output.txt", "test_dir": "tests"}


# ===========================
# Provider calls
# ===========================
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


# ===========================
# Optional chunking (DISABLED by default)
# ===========================
def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 600) -> Iterable[str]:
    """Split sources into overlapping chunks (not used by default)."""
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


# ===========================
# Formatting helpers
# ===========================
def _is_single_liney(s: str) -> bool:
    return ("\n" not in s) or (len(max(s.splitlines(), key=len, default="")) > 300)


def _pretty_format(path: str, content: str) -> str:
    """Heuristics to restore readability if a model returns long/flat blobs."""
    if not _is_single_liney(content):
        return content

    c = content
    # Generic structure
    c = re.sub(r";(?!\s*\n)", ";\n", c)
    c = re.sub(r"\{", "{\n", c)
    c = re.sub(r"\}(?!\s*\n)", "\n}\n", c)
    c = re.sub(r"\)\s*\{", "){\n", c)
    lower = path.lower()
    if lower.endswith(".py"):
        c = c.replace("def ", "\n\ndef ").replace("class ", "\n\nclass ")
    if lower.endswith((".java", ".cs", ".js", ".groovy")):
        c = re.sub(r"(public|private|protected|class|interface|return|if|else|for|while)\s", r"\n\1 ", c)
    if lower.endswith(".sh"):
        c = c.replace(" && ", " && \\\n")
    if lower.endswith((".yml", ".yaml")):
        c = re.sub(r"(\w+:)", r"\n\1", c)

    # Crude brace-based indentation
    lines = c.splitlines()
    out: List[str] = []
    indent = 0
    for line in lines:
        line = line.rstrip()
        if re.search(r"^\s*[\}\)]", line):
            indent = max(0, indent - 1)
        out.append(("    " * indent) + line.lstrip())
        if re.search(r"\{\s*$", line):
            indent += 1
    return "\n".join(out).strip() + "\n"


# ===========================
# Parse <file> and <notes> blocks
# ===========================
_FILE_RE = re.compile(r'<file\s+path="([^"]+)">\s*(.?)\s</file>', re.DOTALL | re.IGNORECASE)
_NOTES_RE = re.compile(r'<notes>\s*(.?)\s</notes>', re.DOTALL | re.IGNORECASE)

def _parse_multifile_response(text: str, fallback_path: str) -> Tuple[List[Dict[str, str]], str]:
    files: List[Dict[str, str]] = []
    notes = ""

    for path, body in _FILE_RE.findall(text):
        # normalize slashes and strip leading ./ if present
        norm_path = re.sub(r"^[./]+", "", path.strip())
        files.append({"path": norm_path, "content": body.strip() + "\n"})

    nm = _NOTES_RE.search(text)
    if nm:
        notes = nm.group(1).strip()

    # Fallback: if no <file> tags, treat entire output as a single file
    if not files:
        files = [{"path": fallback_path, "content": text.strip() + "\n"}]

    return files, notes


# ===========================
# Missing dependency detection
# ===========================
def _detect_missing(legacy_code: str, legacy_type: str) -> List[str]:
    u = legacy_code.upper()
    missing: List[str] = []

    if legacy_type == "cobol" or legacy_type == "copybook":
        # COPY statements
        for m in re.finditer(r"\bCOPY\s+([A-Z0-9\-\._]+)", u):
            missing.append(f"COBOL COPY requires copybook: {m.group(1)}")
        # INCLUDE (DB2 precompiler style)
        for m in re.finditer(r"\bINCLUDE\s+([A-Z0-9\-\._]+)", u):
            missing.append(f"COBOL INCLUDE may require: {m.group(1)}")

    if legacy_type == "jcl":
        # Datasets in DDs
        for m in re.finditer(r"\bDD\s+.*\bDSN=([^,)\s]+)", u):
            missing.append(f"JCL references dataset: DSN={m.group(1)}")
        # In-stream SYSIN, PROC and INCLUDE calls
        if "//SYSIN" in u and "DD *" in u:
            missing.append("JCL has in-stream SYSIN control cards — verify control cards content.")
        for m in re.finditer(r"//\s*([A-Z0-9$#@]+)\s+PROC\b", u):
            missing.append(f"JCL invokes PROC: {m.group(1)} (ensure PROC is available).")
        if "INCLUDE MEMBER=" in u:
            missing.append("JCL INCLUDE MEMBER= detected — ensure included members are provided.")

    if "EXEC CICS" in u:
        missing.append("CICS commands present — ensure transaction & COMMAREA mappings are defined.")
    if "EXEC SQL" in u:
        missing.append("DB2 embedded SQL detected — confirm schema/tables and connection parameters.")

    return missing


# ===========================
# Main conversion API
# ===========================
def convert_to_files(
    legacy_code: str,
    target_stack: str,
    instructions: str,
    requested_artifacts: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str = "OpenAI",
    extra_context: Dict[str, Any] = None,
    enable_chunking: bool = False,  # keep off by default
) -> Dict[str, Any]:
    """
    Convert a single legacy source into modern artifacts.
    Returns: {"files": [{"path","content"},...], "notes": "...", "missing": [...], "usage": {...}, "detected_type": "..."}
    """
    legacy_type = detect_legacy_type(legacy_code)

    # JCL defaults
    if legacy_type == "jcl":
        if "groovy" in target_stack.lower():
            target_stack = "Groovy"
        elif "shell" in target_stack.lower() or "bash" in target_stack.lower():
            target_stack = "Shell Script"
        else:
            target_stack = "Shell Script"

    defaults = _stack_defaults(target_stack)
    fallback_main = defaults["main"]

    # (Chunking disabled unless explicitly enabled)
    chunks = [legacy_code]
    # if enable_chunking:
    #     chunks = list(_chunk_text(legacy_code))

    all_files: List[Dict[str, str]] = []
    all_notes: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        payload = {
            "target_stack": target_stack,
            "instructions": instructions,
            "requested_artifacts": requested_artifacts,
            "legacy_type": legacy_type,
            "parsed_legacy": chunk,
            "defaults": defaults,
        }
        if extra_context:
            payload["context"] = extra_context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]

        raw = _call_groq(messages, model, temperature, max_tokens) if provider.lower() == "groq" \
            else _call_openai(messages, model, temperature, max_tokens)

        files, notes = _parse_multifile_response(raw, fallback_path=fallback_main)
        for f in files:
            f["content"] = _pretty_format(f.get("path", "output.txt"), f.get("content", ""))
        all_files.extend(files)
        if notes:
            all_notes.append(notes)

    missing = _detect_missing(legacy_code, legacy_type)

    return {
        "files": all_files,
        "notes": ("\n\n---\n\n".join(all_notes)).strip(),
        "missing": missing,
        "usage": {"provider": provider, "model": model},
        "detected_type": legacy_type,
    }


# Backward-compatible wrapper used by the app
def convert_to_bundle(*args, **kwargs) -> Dict[str, Any]:
    return convert_to_files(*args, **kwargs)


# ===========================
# Q&A chatbot
# ===========================
def chatbot(
    legacy_code: str,
    query: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    provider: str = "OpenAI"
) -> str:
    legacy_type = detect_legacy_type(legacy_code)
    messages = [
        {"role": "system", "content": f"You are a modernization assistant. Legacy type = {legacy_type}."},
        {"role": "user", "content": f"Legacy code (combined):\n{legacy_code}\n\nUser question:\n{query}\n\nPlease answer clearly and concisely."}
    ]
    return _call_groq(messages, model, temperature, max_tokens) if provider.lower() == "groq" \
        else _call_openai(messages, model, temperature, max_tokens)
