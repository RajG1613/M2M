# converter.py
import os
import time
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from openai._exceptions import APIError, APIConnectionError, RateLimitError

API_KEY = os.getenv("MY_NEW_APP_KEY")
if not API_KEY:
    raise ValueError("Environment variable MY_NEW_APP_KEY is missing. Set it in Render → Environment → MY_NEW_APP_KEY")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a senior Mainframe Modernization architect. "
    "You convert COBOL/JCL into modern, production-grade artifacts. "
    "Follow clean architecture, keep business logic intact, and prefer explicit types and clear error handling. "
    "Never fabricate external interfaces or data fields that do not exist in the legacy code. "
    "When unsure, document assumptions in notes."
)

def _build_user_prompt(legacy_code: str, target_stack: str, instructions: str, requested_artifacts: List[str]) -> str:
    artifacts_hint = ", ".join(requested_artifacts) if requested_artifacts else "Unit Tests, OpenAPI Spec, CI Pipeline (YAML), Dockerfile, Migration Notes"
    return (
        "=== LEGACY CODE START ===\n"
        f"{legacy_code}\n"
        "=== LEGACY CODE END ===\n\n"
        f"Target stack: {target_stack}\n"
        f"Requested artifacts (generate reasonable defaults if not fully inferable): {artifacts_hint}\n"
        f"Additional instructions: {instructions}\n\n"
        "Return a STRICT JSON object with this shape:\n"
        "{\n"
        '  "files": [\n'
        '    {"path": "relative/path/with/extension", "content": "file content as utf-8 text"}\n'
        "  ],\n"
        '  "notes_markdown": "short markdown bullet points about decisions, risks, assumptions"\n'
        "}\n"
        "Rules:\n"
        "- Use realistic project structure for the chosen stack.\n"
        "- For Java/Spring, include a minimal build file (pom.xml or gradle), a main Application class, and a REST controller if applicable.\n"
        "- If tests requested, include them under a conventional test path.\n"
        "- If OpenAPI requested, include openapi.yaml.\n"
        "- If CI requested, include .github/workflows/ci.yml.\n"
        "- If Dockerfile requested, include Dockerfile.\n"
        "- If K8s requested, include k8s/deployment.yml and k8s/service.yml.\n"
        "- Keep file paths portable and do not include absolute paths.\n"
        "- Escape JSON correctly. Do NOT wrap the JSON in backticks.\n"
    )

def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # attempt to salvage JSON if the model added stray text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise e

def convert_to_bundle(
    legacy_code: str,
    target_stack: str,
    instructions: str,
    requested_artifacts: Optional[List[str]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    retries: int = 3,
    backoff: float = 2.0,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "files": [{"path": "...", "content": "..."}, ...],
        "notes_markdown": "…",
        "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
      }
    """
    user_prompt = _build_user_prompt(legacy_code, target_stack, instructions, requested_artifacts or [])

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content
            data = _safe_json_loads(raw)

            # normalize shape
            files = data.get("files", [])
            notes = data.get("notes_markdown", "")

            usage = {}
            try:
                u = getattr(resp, "usage", None)
                if u:
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "total_tokens": getattr(u, "total_tokens", None),
                    }
            except Exception:
                usage = {}

            return {
                "files": files,
                "notes_markdown": notes,
                "usage": usage,
            }
        except (RateLimitError, APIConnectionError, APIError) as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(backoff * attempt)
        except Exception as e:
            # Non-retryable
            raise e

    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_err}")
