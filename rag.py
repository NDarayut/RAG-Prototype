# Langchain dependencies
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
import google.generativeai as genai
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate # Importing prompt template for chat models

load_dotenv()

DATA_PATH = "./data/"
CHROMA_PATH = "chroma"
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

model_name = "TheBloke/Nous-Hermes-13b-GPTQ"  # Example free open model

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

query_text = "When did the trinity test happen?"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
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

def ask_question(query_text: str, k: int = 3):
    """
    Ask a question using the vector store and return the answer and matched context metadata.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    results = db.similarity_search(query_text, k=k)

    context_chunks = []
    for doc in results:
        metadata = doc.metadata or {}
        context_chunks.append({
            "filename": os.path.basename(metadata.get("source", "unknown.pdf")),
            "page": metadata.get("page", 1),  # Ensure this is 1-indexed
            "text": doc.page_content.strip()
        })

    context_text = "\n\n".join([doc.page_content for doc in results])

    # Format prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    # Use Gemini model directly (no Langchain wrapper)
    response = gemini_model.generate_content(prompt)

    return response.text.strip(), context_chunks

