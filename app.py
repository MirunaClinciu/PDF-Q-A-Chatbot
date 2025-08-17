from __future__ import annotations

import os
import io
import logging
from typing import List, Tuple, Dict

import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__, template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdf_chatbot")


class Settings:
    EMBED_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo" # You can change upgrade your model but this one is ok too.
    OPENAI_API_KEY = "YOUR OPENAI API" # You need to update this!!!!
    MAX_CONTENT_LENGTH_MB = 25
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    EMBED_BATCH = 32
    TOP_K = 5
    ALLOWED_EXTENSIONS = {".pdf"}


SET = Settings()
app.config["MAX_CONTENT_LENGTH"] = SET.MAX_CONTENT_LENGTH_MB * 1024 * 1024

log.info("Loading embedding model: %s", SET.EMBED_MODEL)
log.info("Using OpenAI model: %s", SET.LLM_MODEL)

embedder = HuggingFaceEmbeddings(
    model_name=SET.EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": SET.EMBED_BATCH, "normalize_embeddings": True},
)

vector_store = None
index_ready = False


def is_allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SET.ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes), strict=False)
    return [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]


def build_index(pages: List[Tuple[int, str]]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=SET.CHUNK_SIZE, chunk_overlap=SET.CHUNK_OVERLAP)
    texts, metas = [], []
    for page_num, text in pages:
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            texts.append(chunk)
            metas.append({
                "page": page_num,
                "source": f"Page {page_num}"
            })
    if not texts:
        raise ValueError("No valid content found.")
    return FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)


def generate_answer(query: str, top_k=5) -> Dict:
    from langchain_openai import ChatOpenAI

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(
        model=SET.LLM_MODEL,
        temperature=0,
        api_key=SET.OPENAI_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context only. Say 'I don't know' if unsure."),
        ("human", "Q: {input}\n\nContext:\n{context}")
    ])
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    )
    result = chain.invoke({"input": query})

    answer = result.get("answer") or result.get("output_text") or ""
    source_pages = sorted({
        f"Page {doc.metadata.get('page', '?')}"
        for doc in result.get("context", [])
        if "page" in doc.metadata
    })
    sources = ", ".join(source_pages)

    return {"answer": answer.strip(), "sources": sources or "Unknown"}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global vector_store, index_ready

    if "file" not in request.files:
        return jsonify(ok=False, error="No file uploaded."), 400

    file = request.files["file"]
    if not file or not is_allowed_file(file.filename):
        return jsonify(ok=False, error="Invalid file type."), 400

    try:
        file_bytes = file.read()
        pages = extract_text_from_pdf(file_bytes)
        if not pages:
            return jsonify(ok=False, error="No text extracted."), 400

        vector_store = build_index(pages)
        index_ready = True
        return jsonify(ok=True, pages_indexed=len(pages))

    except PdfReadError as e:
        log.exception("PDF parsing error")
        return jsonify(ok=False, error=f"PDF parsing failed: {e}"), 400
    except Exception as e:
        log.exception("Upload failure")
        return jsonify(ok=False, error=f"Upload failed: {e}"), 500


@app.route("/chat", methods=["POST"])
def chat():
    if not index_ready or vector_store is None:
        return jsonify(ok=False, error="No PDF indexed. Please upload one first."), 400

    data = request.get_json(force=True) or {}
    query = data.get("message", "").strip()
    if not query:
        return jsonify(ok=False, error="Missing 'message'."), 400

    try:
        k = int(data.get("k", SET.TOP_K))
        result = generate_answer(query, k)
        return jsonify(ok=True, **result)
    except Exception as e:
        log.exception("Chat failure")
        return jsonify(ok=False, error=f"Chat error: {e}"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
