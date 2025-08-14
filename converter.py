# converter.py
import os
from openai import OpenAI

# Load API key from environment variable MY_NEW_APP_KEY
api_key = os.getenv("MY_NEW_APP_KEY")

if not api_key:
    raise ValueError("❌ API key not found! Please set MY_NEW_APP_KEY in your environment variables.")

# Create OpenAI client
client = OpenAI(api_key=api_key)

def convert_code(code: str, prompt: str):
    """
    Convert legacy mainframe code (COBOL/JCL) to modern languages using GPT-4o.
    """
    system_msg = (
        "You are an expert in Mainframe Modernization. "
        "Convert COBOL/JCL to modern languages with best practices."
    )
    user_msg = f"Legacy Code:\n{code}\n\nUser Prompt:\n{prompt}"

    # Call OpenAI Chat API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    return response.choices[0].message.content
