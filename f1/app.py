import logging
import os
import io
import urllib.parse
import streamlit as st
from model_serving_utils import query_endpoint
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Configure logging\logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Raw env var (can be full URL or just name)
RAW_SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")

if not RAW_SERVING_ENDPOINT:
    st.error(
        "❗ The SERVING_ENDPOINT environment variable is not set.\n"
        "Please add it to your .env or app.yaml."
    )
    st.stop()

# Parse the endpoint name if a full URL is given
def parse_endpoint_name(raw: str) -> str:
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        segments = parsed.path.strip("/").split("/")
        # Find the segment after 'serving-endpoints'
        try:
            idx = segments.index("serving-endpoints")
            return segments[idx + 1]
        except (ValueError, IndexError):
            logger.warning("Could not parse endpoint name from URL; using raw value.")
            return raw
    return raw

SERVING_ENDPOINT_NAME = parse_endpoint_name(RAW_SERVING_ENDPOINT)
logger.info(f"Using serving endpoint name: {SERVING_ENDPOINT_NAME}")

# Extract user info (safe stub)
def get_user_info():
    try:
        headers = st.context.headers
    except Exception:
        headers = {}
    return {
        "user_name": headers.get("X-Forwarded-Preferred-Username"),
        "user_email": headers.get("X-Forwarded-Email"),
        "user_id": headers.get("X-Forwarded-User"),
    }

user_info = get_user_info()

# Streamlit UI setup
st.set_page_config(page_title="📄 Asideus Chatbot - beta")
st.title("📄 Asideus Chatbot - beta")
st.markdown("Upload PDF or DOCX documents and ask questions based on their contents.")

# Session state defaults
st.session_state.setdefault("messages", [])
st.session_state.setdefault("docs_text", "")

# File uploader
uploaded = st.file_uploader(
    "Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True
)
if uploaded:
    extracted = []
    for file in uploaded:
        data = file.read()
        if file.type == "application/pdf":
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            doc = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
        extracted.append(f"--- {file.name} ---\n{text}")
    st.session_state.docs_text = "\n\n".join(extracted)
    st.success("✅ Documents processed.")

# Render previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything based on your documents...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build payload
    to_model = []
    if st.session_state["docs_text"]:
        to_model.append({"role": "system", "content": st.session_state["docs_text"]})
    to_model.extend(st.session_state["messages"])

    # Call MLflow/Databricks
    with st.chat_message("assistant"):
        resp = query_endpoint(
            endpoint_name=SERVING_ENDPOINT_NAME,
            messages=to_model,
            max_tokens=400,
        )
        ans = resp.get("content") if isinstance(resp, dict) else resp
        st.markdown(ans)
    st.session_state["messages"].append({"role": "assistant", "content": ans})
