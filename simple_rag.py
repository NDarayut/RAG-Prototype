# Modified RAG Pipeline for General Document Q&A (Khmer & English)

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader

logging.basicConfig(level=logging.INFO)

use_gpu = torch.cuda.is_available()
model_id = "aisingapore/Llama-SEA-LION-v3.5-8B-R"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_8bit=True,          # Enable 8-bit quantization
    torch_dtype=torch.float16,
)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

DATA_PATH = "./data/"
CHROMA_PATH = "chroma"
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Generic assistant prompt for dual Khmer/English
PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer the question based ONLY on the context below.
If the user asks in Khmer, respond in Khmer.
If the user asks in English, respond in English.
Use clear, concise sentences. Do not mention the existence of context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        db.add_documents(chunks)
        logging.info("Added documents to existing Chroma DB.")
    else:
        db = Chroma.from_documents(
            chunks, embedding_model, persist_directory=CHROMA_PATH
        )
        logging.info("Created new Chroma DB.")
    db.persist()
    logging.info(f"Saved {len(chunks)} chunks to Chroma.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def ask_question(query_text: str, k: int = 3):
    logging.info("Processing user question...")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    results = db.similarity_search(query_text, k=k)

    context_chunks = []
    for doc in results:
        meta = doc.metadata or {}
        context_chunks.append({
            "filename": os.path.basename(meta.get("source", "unknown.pdf")),
            "page": meta.get("page", 1),
            "text": doc.page_content.strip()
        })

    context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    messages = [{"role": "user", "content": prompt}]
    logging.info("Sending prompt to model...")
    prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            thinking_mode="off"
        )

    output = pipeline(
        prompt,
        max_new_tokens=1024,
        return_full_text=False,
        truncation=True,
        do_sample=False,
    )

    answer = output[0]["generated_text"].strip()
    return answer, context_chunks
