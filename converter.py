# converter.py
import openai

# 🔹 Replace with your real OpenAI API key
openai.api_key = "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"

def convert_code(code: str, prompt: str):
    system_msg = (
        "You are an expert in Mainframe Modernization. "
        "Convert COBOL/JCL to modern languages with best practices."
    )
    user_msg = f"Legacy Code:\n{code}\n\nUser Prompt:\n{prompt}"

    # Call OpenAI Chat API
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content

