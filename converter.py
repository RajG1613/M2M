# converter.py
import os
import json
import re
from typing import Dict, List, Any

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
   * By default => convert to Shell scripts (bash) or CI/CD YAML.
   * If target stack contains 'groovy' => convert to Groovy (Jenkins pipeline style).
- COBOL => convert into the requested target stack (Spring/Java, FastAPI/Python, .NET, Node).
- COPYBOOK => convert into DTO/POJO classes, schema files, or reusable modules in the target stack.
- DB2 => modern SQL/ORM.
- VSAM => relational/NoSQL schema + data access code.
- CICS => REST APIs (controller + service layer) in the chosen stack.
- IMS DB => RDBMS schema + migration utilities.

Formatting Rules:
- Output must be well-formatted (line breaks, indentation).
- Keep idiomatic conventions for the chosen stack.
- Each COBOL program, JCL member, and Copybook should map to a separate output file.

Respond STRICTLY as JSON **only** with this schema:
{
  "files": [
    {"path": "relative/path/with/extension", "content": "file text"},
    ...
  ],
  "notes_markdown": "bullet points of migration decisions, risks, assumptions",
  "usage": {
    "notes": "optional",
    "hints": "optional"
  }
}
"""


# -------- Helpers --------
def _stack_defaults(target_stack: str) -> Dict[str, str]:
    ts = target_stack.lower()
    if "spring" in ts:  # Java/Spring Boot
        return {"main": "src/main/java/com/example/App.java", "test_dir": "src/test/java/com/example"}
    if "fastapi" in ts:  # Python
        return {"main": "app/main.py", "test_dir": "tests"}
    if ".net" in ts or "c#" in ts:
        return {"main": "src/Program.cs", "test_dir": "tests"}
    if "node" in ts or "express" in ts:
        return {"main": "src/index.js", "test_dir": "__tests__"}
    if "groovy" in ts:  # JCL -> Groovy
        return {"main": "jenkins/Jenkinsfile.groovy", "test_dir": "tests"}
    if "shell" in ts:  # JCL -> shell
        return {"main": "scripts/job.sh", "test_dir": "tests"}
    return {"main": "src/output.txt", "test_dir": "tests"}


def _detect_legacy_type(legacy_code: str) -> str:
    code_upper = legacy_code.upper()
    if code_upper.strip().startswith("//") or " JOB " in code_upper:
        return "jcl"
    if "PROCEDURE DIVISION" in code_upper:
        return "cobol"
    if " COPY " in code_upper or code_upper.strip().startswith("01 "):
        return "copybook"
    if "EXEC SQL" in code_upper:
        return "db2"
    if "CICS" in code_upper:
        return "cics"
    if "IMS" in code_upper:
        return "ims"
    return "unknown"


def _chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """Split legacy code into chunks for large files."""
    lines = text.splitlines()
    chunks, current = [], []
    total_len = 0
    for line in lines:
        total_len += len(line) + 1
        if total_len > max_chunk_size:
            chunks.append("\n".join(current))
            current, total_len = [], len(line)
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _call_openai(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("MY_NEW_APP_KEY")
    if not api_key:
        raise RuntimeError("OpenAI key not found: set env MY_NEW_APP_KEY")
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Add `openai` to requirements.txt")
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
        raise RuntimeError("groq SDK not installed. Add `groq` to requirements.txt")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content


# -------- Main Converter --------
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
    Convert legacy code into modern artifacts with chunking and multi-file support.
    """
    legacy_type = _detect_legacy_type(legacy_code)

    # Override JCL defaults
    if legacy_type == "jcl":
        if "groovy" in target_stack.lower():
            target_stack = "Groovy"
        elif "shell" in target_stack.lower():
            target_stack = "Shell Script"
        else:
            target_stack = "Shell Script"

    defaults = _stack_defaults(target_stack)

    # Chunk legacy code
    chunks = _chunk_text(legacy_code)

    files, notes = [], []
    for idx, chunk in enumerate(chunks):
        user_payload = {
            "target_stack": target_stack,
            "instructions": instructions,
            "requested_artifacts": requested_artifacts,
            "parsed_legacy": chunk,
            "legacy_type": legacy_type,
            "defaults": defaults,
        }
        if extra_context:
            user_payload["context"] = extra_context

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

        if provider.lower() == "groq":
            content = _call_groq(messages, model, temperature, max_tokens)
        else:
            content = _call_openai(messages, model, temperature, max_tokens)

        try:
            bundle = json.loads(content)
            files.extend(bundle.get("files", []))
            if "notes_markdown" in bundle:
                notes.append(bundle["notes_markdown"])
        except Exception:
            files.append({"path": f"{defaults['main']}.part{idx+1}", "content": content})
            notes.append("⚠️ Model did not return JSON for one chunk.")

    final_bundle = {
        "files": files,
        "notes_markdown": "\n\n".join(notes),
        "usage": {"provider": provider, "model": model},
    }
    return final_bundle


# -------- Chatbot Mode --------
def chatbot(
    legacy_code: str,
    query: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    provider: str = "OpenAI"
) -> str:
    """
    Interactive chatbot for modernization Q&A.
    """
    legacy_type = _detect_legacy_type(legacy_code)
    messages = [
        {"role": "system", "content": f"You are a modernization assistant. Legacy type = {legacy_type}."},
        {"role": "user", "content": f"Legacy code:\n{legacy_code}\n\nUser question:\n{query}"}
    ]

    if provider.lower() == "groq":
        return _call_groq(messages, model, temperature, max_tokens)
    else:
        return _call_openai(messages, model, temperature, max_tokens)
