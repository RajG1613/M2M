# converter.py
from groq import Groq
from openai import OpenAI

def convert_to_bundle(legacy_code, target_stack, instructions, requested_artifacts, model, temperature, max_tokens, provider="OpenAI", api_key=None):
    if provider == "Groq":
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a code modernization assistant."},
                      {"role": "user", "content": f"Code:\n{legacy_code}\n\nInstructions:\n{instructions}"}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        output = resp.choices[0].message.content
    else:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a code modernization assistant."},
                      {"role": "user", "content": f"Code:\n{legacy_code}\n\nInstructions:\n{instructions}"}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        output = resp.choices[0].message.content

    return {
        "files": [{"path": f"src/output.{target_stack.lower()[:2]}", "content": output}],
        "notes_markdown": "Modernization completed.",
        "usage": {"provider": provider, "model": model}
    }
