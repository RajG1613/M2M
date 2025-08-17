# converter.py
import os
import io
import json
import zipfile
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


SYSTEM_PROMPT = """You are an expert in Mainframe Modernization.
You will receive:
- Parsed legacy code (COBOL/JCL/DB2/VSAM/CICS/IMS DB) with filename context
- Target stack and user instructions
- Requested artifacts to include

RESPOND STRICTLY AS JSON with this schema:
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

Guidelines:
- Always preserve original business logic while modernizing.
- JCL ➝ Convert to shell scripts (bash) or modern CI/CD YAML.
- DB2 ➝ Convert SQL to Postgres/MySQL or ORM equivalents.
- VSAM ➝ Convert to relational tables or JSON/NoSQL.
- CICS ➝ Convert to REST APIs (Spring Boot / FastAPI / .NET Web API).
- IMS DB ➝ Convert to relational DB schemas + migration utilities.
- Generate converted code **line by line, professionally formatted**.
- Unit tests must be in a separate document under /tests.
- If 'OpenAPI Spec' requested, include swagger/openapi file under /api.
- If 'CI Pipeline (YAML)' requested, include a YAML pipeline file.
- If 'Dockerfile' requested, include root-level Dockerfile.
- If 'K8s Manifests' requested, include basic deployment/service YAML.
- Always include comments where business logic is inferred.
"""


def _stack_defaults(target_stack: str) -> Dict[str, str]:
    ts = target_stack.lower()
    if "spring" in ts:
        return {"main": "src/main/java/App.java", "test_dir": "src/test/java"}
    if "fastapi" in ts:
        return {"main": "app/main.py", "test_dir": "tests"}
    if ".net" in ts or "c#" in ts:
        return {"main": "src/Program.cs", "test_dir": "tests"}
    if "node" in ts or "express" in ts:
        return {"main": "src/index.js", "test_dir": "__tests__"}
    return {"main": "src/output.txt", "test_dir": "tests"}


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
    zip_output: bool = True,
) -> Dict[str, Any]:
    """
    Returns dict:
    {
      "files": [{"path": "...", "content": "..."}, ...],
      "notes_markdown": "...",
      "usage": {"provider": "...", "model": "..."},
      "zip_bytes": optional zip archive as bytes
    }
    """
    defaults = _stack_defaults(target_stack)

    user_payload = {
        "target_stack": target_stack,
        "instructions": instructions,
        "requested_artifacts": requested_artifacts,
        "parsed_legacy": legacy_code,
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
            raise ValueError("No 'files' field in JSON")
    except Exception:
        bundle = {
            "files": [{"path": defaults["main"], "content": content}],
            "notes_markdown": "Model returned non-JSON text. Wrapped into a single file.",
            "usage": {}
        }

    # usage meta
    usage = bundle.get("usage", {})
    usage.update({"provider": provider, "model": model})
    bundle["usage"] = usage

    # prepare zip if requested
    if zip_output:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in bundle["files"]:
                zf.writestr(f["path"], f["content"])
            zf.writestr("NOTES.md", bundle.get("notes_markdown", ""))
        bundle["zip_bytes"] = zip_buffer.getvalue()

    return bundle
