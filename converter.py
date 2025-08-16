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


SYSTEM_PROMPT = """You are an expert in Mainframe Modernization.
You will receive:
- Parsed legacy code (COBOL/JCL) with filename context
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
- Use production-quality patterns for the chosen stack.
- If 'Unit Tests' requested, include them under a conventional test folder.
- If 'OpenAPI Spec' requested, include swagger/openapi file under /api.
- If 'CI Pipeline (YAML)' requested, include a YAML pipeline file.
- If 'Dockerfile' requested, include a root-level Dockerfile.
- If 'K8s Manifests' requested, include basic deployment/service YAML.
- Always include comments where business logic is inferred.
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
) -> Dict[str, Any]:
    """
    Returns dict:
    {
      "files": [{"path": "...", "content": "..."}, ...],
      "notes_markdown": "...",
      "usage": {"provider": "...", "model": "...", ...}
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

    # Try to parse strict JSON; if model returned plain text, wrap it.
    bundle: Dict[str, Any]
    try:
        bundle = json.loads(content)
        if "files" not in bundle:
            raise ValueError("No 'files' field in JSON")
    except Exception:
        # Fallback: deliver single main file with whatever the model returned
        bundle = {
            "files": [{"path": defaults["main"], "content": content}],
            "notes_markdown": "Model returned non-JSON text. Wrapped into a single file.",
            "usage": {}
        }

    # Add usage meta
    usage = bundle.get("usage", {})
    usage.update({"provider": provider, "model": model})
    bundle["usage"] = usage
    return bundle
