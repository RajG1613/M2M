# app.py
import io
import os
import time
import zipfile
import streamlit as st
from typing import List, Dict
from converter import convert_to_bundle
from parser import parse_code

# --- NEW: Import Groq client ---
from groq import Groq
from openai import OpenAI

st.set_page_config(page_title="AI Mainframe Modernizer", page_icon="🚀", layout="wide")

st.title("🚀 AI Mainframe Modernizer — Enterprise Demo")
st.caption("Upload COBOL/JCL → get production-grade modern artifacts.")

with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("Provider", ["OpenAI", "Groq"])
    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        api_key = os.getenv("MY_NEW_APP_KEY")
    else:
        model = st.selectbox("Model", ["mixtral-8x7b-32768", "llama-3.1-70b-versatile"])
        api_key = os.getenv("GROQ_API_KEY")

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max output tokens", min_value=512, max_value=8192, value=4096, step=256)

    st.divider()
    target_stack = st.selectbox(
        "Target stack",
        ["Java (Spring Boot)", "Python (FastAPI)", "C# (.NET)", "Node.js (Express)"],
    )
    extras = st.multiselect(
        "Artifacts to include",
        ["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "K8s Manifests", "Migration Notes"],
        default=["Unit Tests", "OpenAPI Spec", "CI Pipeline (YAML)", "Dockerfile", "Migration Notes"]
    )
    st.divider()
    st.caption("🔐 API keys read from env vars: **MY_NEW_APP_KEY** (OpenAI), **GROQ_API_KEY** (Groq).")

# rest of your file uploader + button remains the same
# BUT update convert_to_bundle to accept provider + api_key
