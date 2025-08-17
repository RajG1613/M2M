# converter.py
import os
import json
from typing import Dict, List, Any

# OpenAI SDK (modern)
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
- If legacy type is JCL => ALWAYS convert to shell scripts (bash) or CI/CD YAML,
  even if target stack is Spring Boot, FastAPI, etc.
- COBOL => convert into the requested target stack (Spring/Java, FastAPI/Python, etc.)
- DB2 => modern SQL/ORM
- VSAM => relational/NoSQL schema + data access code
- CICS => REST APIs (controller + service layer)
- IMS DB => RDBMS schema + migration utilities

Respond STRICTLY as JSON with this schema:
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


def _stack_defaults(target_stack: str) -> Dict[str, str]:
    ts = target_stack.lower()
    if "spring" in ts:  # Java/Spring Boot
        return {"main": "src/main/java/App.java", "test_dir": "src/test/java"}
    if "fastapi" in ts:  # Python
        return {"main": "app/main.py", "test_dir": "tests"}
    if ".net" in ts or "c#" in ts:
        return {"main": "src/Program.cs", "test_dir": "tests"}
    if "node" in ts or "express" in ts:
        return {"main": "src/index.js", "test_dir": "__tests__"}
    if "shell" in ts:  # For JCL → shell
        return {"main": "scripts/job.sh", "test_dir": "tests"}
    return {"main": "src/output.txt", "test_dir": "tests"}


def _detect_legacy_type(legacy_code: str) -> str:
    code_upper = legacy_code.upper()
    if code_upper.strip().startswith("//") or " JOB " in code_upper:
        return "jcl"
    if "PROCEDURE DIVISION" in code_upper:
        return "cobol"
    if "EXEC SQL" in code_upper:
        return "db2"
    if "CICS" in code_upper:
        return "cics"
    if "IMS" in code_upper:
        return "ims"
    return "unknown"


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


def convert_to_bundle(
    legacy_code: str,
    target_stack: str,
    instructions: str,
    requested_artifacts: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str = "OpenAI",
) -> Dict[str, Any]:
    """
    Convert legacy code into modern artifacts.
    """
    legacy_type = _detect_legacy_type(legacy_code)

    # Override if JCL
    if legacy_type == "jcl":
        target_stack = "Shell Script"

    defaults = _stack_defaults(target_stack)

    user_payload = {
        "target_stack": target_stack,
        "instructions": instructions,
        "requested_artifacts": requested_artifacts,
        "parsed_legacy": legacy_code,
        "legacy_type": legacy_type,
        "defaults": defaults,
    }

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
        if "files" not in bundle:
            raise ValueError("No 'files' in JSON")
    except Exception:
        bundle = {
            "files": [{"path": defaults["main"], "content": content}],
            "notes_markdown": "⚠️ Model did not return JSON. Wrapped raw output.",
            "usage": {}
        }

    usage = bundle.get("usage", {})
    usage.update({"provider": provider, "model": model})
    bundle["usage"] = usage
    return bundle


# -------- Interactive Chatbot --------

def chatbot(legacy_code: str, query: str, model: str, temperature: float = 0.2, max_tokens: int = 1000, provider: str = "OpenAI") -> str:
    """
    Interactive chatbot for modernization Q&A.
    Example: explain COBOL logic, map DB2 to ORM, etc.
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
