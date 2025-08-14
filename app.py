
from fastapi import FastAPI, UploadFile, Form
from converter import convert_code
from parser import parse_code
import streamlit as st
import requests
import threading
import uvicorn

app = FastAPI(title="AI Mainframe Modernizer")

@app.post("/convert")
async def convert_mainframe(file: UploadFile, prompt: str = Form(...)):
    # 1. Read file
    content = await file.read()
    legacy_code = content.decode("utf-8")

    # 2. Parse legacy code
    structured_code = parse_code(legacy_code)

    # 3. Convert using AI
    modern_code = convert_code(structured_code, prompt)

    return {"converted_code": modern_code}

def run_streamlit():
    st.title("🚀 AI Mainframe Modernizer")
    st.write("Upload COBOL/JCL and get modernized code instantly!")

    uploaded_file = st.file_uploader("Upload Legacy File", type=["txt","cbl","jcl"])
    prompt = st.text_area("Enter your modernization prompt",
                          "Convert this COBOL program to Java with Spring Boot")

    if st.button("Modernize Code") and uploaded_file:
        files = {'file': uploaded_file.getvalue()}
        data = {'prompt': prompt}

        res = requests.post("http://localhost:8000/convert", files=files, data=data)
        if res.status_code == 200:
            modern_code = res.json()["converted_code"]
            st.code(modern_code, language='java')
            st.download_button("Download Modern Code", modern_code)

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api)
    api_thread.start()
    run_streamlit()