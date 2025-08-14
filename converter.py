# converter.py
import os
from openai import OpenAI

# Create OpenAI client with API key from environment
client = OpenAI(api_key=os.environ["MY_NEW_APP_KEY"])

def convert_code(code: str, prompt: str):
    system_msg = (
        "You are an expert in Mainframe Modernization. "
        "Convert COBOL/JCL to modern languages with best practices."
    )
    user_msg = f"Legacy Code:\n{code}\n\nUser Prompt:\n{prompt}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    return response.choices[0].message.content
