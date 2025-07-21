# Langchain dependencies

from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
import google.generativeai as genai
import os # Importing os module for operating system functionalities
from langchain.embeddings import HuggingFaceEmbeddings
import re

load_dotenv()

DATA_PATH = "./data/"
CHROMA_PATH = "chroma"
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT_TEMPLATE = """
You are a helpful loan officer. Match the user's profile against the available loan products listed in the context.
Clearly state which products they qualify for, and explain **why** in simple terms with proper spacing and formatting, no need italicize, just normal text.

User Info:
Location: {location}
Monthly Revenue: ${monthly_income}
Business Age: {business_age_months} months
Collateral: {collateral}
Existing Loans: {existing_loans}

Loan Brochure:
{context}

Respond clearly and in full sentences no need italicize, just normal text:
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

def extract_entities(text: str):
    entities = {
        "location": None,
        "monthly_income": None,
        "business_age_months": None,
        "collateral": None,
        "existing_loans": None
    }

    location_match = re.search(r"in\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    income_match = re.search(r"\$?(\d{3,5})\s?(?:usd|dollars|a month)", text.lower())
    age_match = re.search(r"operating for\s(\d{1,3})\smonths?", text)
    collateral_match = re.search(r"have a\s(.+?)\sas collateral", text)
    loan_match = re.search(r"(no|not have|don't have).*loan", text.lower())

    if location_match:
        entities["location"] = location_match.group(1)
    if income_match:
        entities["monthly_income"] = int(income_match.group(1))
    if age_match:
        entities["business_age_months"] = int(age_match.group(1))
    if collateral_match:
        entities["collateral"] = collateral_match.group(1)
    if loan_match:
        entities["existing_loans"] = "no"

    return entities


def ask_question(query_text: str, k: int = 3):
    from chromadb.utils import embedding_functions
    generate_data_store()  # optional: you may want to skip this call on every query for speed

    # Step 1: NER to extract structured input
    entities = extract_entities(query_text)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    # Step 2: Semantic search for relevant brochure content
    results = db.similarity_search(query_text, k=k)

    # Prepare context chunks for Streamlit display (filename, page, text)
    context_chunks = []
    for doc in results:
        metadata = doc.metadata or {}
        context_chunks.append({
            "filename": os.path.basename(metadata.get("source", "unknown.pdf")),
            "page": metadata.get("page", 1),  # Default to page 1 if missing
            "text": doc.page_content.strip()
        })

    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

    # Step 3: Build final prompt with extracted entities + retrieved context
    prompt = PROMPT_TEMPLATE.format(
        location=entities.get("location", "Unknown"),
        monthly_income=entities.get("monthly_income", "Unknown"),
        business_age_months=entities.get("business_age_months", "Unknown"),
        collateral=entities.get("collateral", "Unknown"),
        existing_loans=entities.get("existing_loans", "Unknown"),
        context=context_text
    )

    response = gemini_model.generate_content(prompt)
    answer = response.text.strip()

    # Return answer and context chunks to display in Streamlit
    return answer, context_chunks



# user_query = "I run a bakery in Phnom Penh, earning about $900 a month. My business has been operating for 14 months. I have a motorcycle as collateral but no existing loans."
# answer, extracted = ask_question(user_query)
# print(answer)

