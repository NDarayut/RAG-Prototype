# Langchain dependencies

import os
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

use_gpu = torch.cuda.is_available()

if use_gpu:
    print("CUDA device in use:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU. No GPU detected.")


model_id = "aisingapore/Llama-SEA-LION-v3.5-8B-R"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if use_gpu:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )


pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if use_gpu else -1,  # -1 means CPU
)

logging.info("Pipeline created.")


DATA_PATH = "./data/"
CHROMA_PATH = "chroma"
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

PROMPT_TEMPLATE = """
You are a helpful loan officer. Your job is to recommend suitable loans based **only** on the customer profile and the available loan products below. Do **not** use any outside knowledge.

Always respond in Khmer if the context is in Khmer. Use short, clear sentences. Do not summarize the context or mention its existence.

Customer Profile:
- Location: {location}
- Monthly Income: ${monthly_income}
- Business Age: {business_age_months} months
- Collateral: {collateral}
- Existing Loans: {existing_loans}

Available Loan Products:
{context}

Instructions:
- List the loan products the customer qualifies for.
- Briefly explain the reason for each match.
- If none qualify, explain why.
- Use only information from the "Available Loan Products" section.
- Never invent or assume anything not stated in the context.
"""



def load_documents():
  """
  Load PDF documents from the specified directory using PyPDFDirectoryLoader.

  Returns:
      List[Document]: Loaded PDF documents with metadata including source file and page number.
  """
  from langchain.document_loaders import PyPDFDirectoryLoader

  # Load PDF documents
  document_loader = PyPDFDirectoryLoader(DATA_PATH)
  documents = document_loader.load()

  # Ensure metadata contains 'source' and 'page'
  for i, doc in enumerate(documents[:5]):  # Just print first 5 for debug
      print(f"Doc {i + 1}:")
      print(f"  Source: {doc.metadata.get('source')}")
      print(f"  Page: {doc.metadata.get('page')}")

  return documents


def split_text(documents: list[Document]):
  """
  Split the text of the loaded documents into smaller chunks using RecursiveCharacterTextSplitter.

  Args:
      documents (list[Document]): List of Document objects to be split.

  Returns:
      List of Document objects: Documents with text split into smaller chunks.
  """
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=512,
      chunk_overlap=100,
      length_function=len,
      add_start_index=True
  )

  chunks = text_splitter.split_documents(documents)

  # Log chunks and metadata for debugging
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
  for i, chunk in enumerate(chunks[:3]):  # Show a few samples
      print(f"Chunk {i + 1} metadata:", chunk.metadata)

  return chunks



def save_to_chroma(chunks: list[Document]):

  """
  Save the split document chunks to a Chroma vector store.
  Args:
      chunks (list[Document]): List of Document objects to be saved.

  Returns:
      None
  """

  # Load existing DB or create a new one
  if os.path.exists(CHROMA_PATH):
      db = Chroma(
          persist_directory=CHROMA_PATH,
          embedding_function=embedding_model
      )
      db.add_documents(chunks)  # Add new documents
      print("Added new documents to existing Chroma vector store.")
  else:
      db = Chroma.from_documents(
          chunks,
          embedding_model,
          persist_directory=CHROMA_PATH
      )
      print("Created new Chroma vector store.")

  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  """
  Main function to generate vector database in chroma from documents.
  Returns:
      None
  """
  documents = load_documents() # Load documents from a source
  chunks = split_text(documents) # Split documents into manageable chunks
  save_to_chroma(chunks) # Save the processed data to a data store

def extract_entities(text: str) -> dict:
    print(f"[INFO] Extracting entities from text: {text}")

    ner_prompt = f"""
    You are an information extractor. Your task is to return ONLY a valid JSON object with these keys: location, monthly_income, business_age_months, collateral, existing_loans.
    
    DO NOT include any explanations, introductions, or any text outside the JSON. ONLY output the JSON.
    
    User input:
    {text}
    
    Output:
    """

    messages = [{"role": "user", "content": ner_prompt}]

    print("[INFO] Running SEA-LION NER prompt with thinking_mode off...")
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            thinking_mode="off"
        )

        output = pipeline(
            prompt,
            max_new_tokens=512,
            return_full_text=False,
            truncation=True,
            do_sample=False,
        )
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return {
            "location": None,
            "monthly_income": None,
            "business_age_months": None,
            "collateral": None,
            "existing_loans": None
        }

    ner_text = output[0].get("generated_text", "").strip()
    print(f"[INFO] Raw NER output:\n{ner_text}")

    entities = {
        "location": None,
        "monthly_income": None,
        "business_age_months": None,
        "collateral": None,
        "existing_loans": None
    }

    

    json_match = re.search(r"\{.*\}", ner_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        print(f"[DEBUG] Matched JSON string: {json_str}")
        try:
            parsed = json.loads(json_str)
            for key in entities.keys():
                if key in parsed:
                    val = parsed[key]
                    if key in ["monthly_income", "business_age_months"]:
                        try:
                            val_clean = int(re.sub(r"[^\d]", "", str(val)))
                            print(f"[DEBUG] Parsed {key}: raw='{val}', cleaned={val_clean}")
                            val = val_clean
                        except Exception as e:
                            print(f"[WARN] Failed to parse {key} value '{val}': {e}")
                    entities[key] = val
                else:
                    print(f"[WARN] Missing key '{key}' in parsed output")
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decoding failed: {e}")
    else:
        print("[WARN] No JSON object found in NER output")

    print(f"[INFO] Extracted entities: {entities}")
    return entities




def ask_question(query_text: str, k: int = 3):
    logging.info("Starting question processing...")

    # Extract entities from user input via SEA-LION NER
    entities = extract_entities(query_text)
    logging.info(f"Extracted entities: {entities}")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    logging.info("Performing semantic search...")
    results = db.similarity_search(query_text, k=k)
    logging.info(f"Retrieved {len(results)} relevant chunks.")

    context_chunks = []
    for doc in results:
        metadata = doc.metadata or {}
        context_chunks.append({
            "filename": os.path.basename(metadata.get("source", "unknown.pdf")),
            "page": metadata.get("page", 1),
            "text": doc.page_content.strip()
        })

    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

    prompt = PROMPT_TEMPLATE.format(
        location=entities.get("location", "Unknown"),
        monthly_income=entities.get("monthly_income", "Unknown"),
        business_age_months=entities.get("business_age_months", "Unknown"),
        collateral=entities.get("collateral", "Unknown"),
        existing_loans=entities.get("existing_loans", "Unknown"),
        context=context_text
    ).strip()

    logging.info("Generating response from SEA-LION pipeline...")
    messages = [{"role": "user", "content": prompt}]

    print("[INFO] Prompt: ", prompt)
    output = pipeline(
        messages,
        max_new_tokens=4096,
        return_full_text=False,
        truncation=True,
    )
    logging.info("Response generation complete.")

    print("[INFO] Output: ", output)

    answer = output[0]["generated_text"].strip()

    return answer, context_chunks
