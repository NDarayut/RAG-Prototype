# RAG Prototype

![GitHub last commit](https://img.shields.io/github/last-commit/NDarayut/RAG-Prototype?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/NDarayut/RAG-Prototype?style=for-the-badge)
![GitHub top language](https://img.shields.io/github/languages/top/NDarayut/RAG-Prototype?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About The Project

This repository hosts a prototype of a Retrieval-Augmented Generation (RAG) application. The primary goal of this project is to demonstrate how to build a system that allows users to interact with their documents by asking questions and receiving answers grounded in the content of those documents.

The application enables users to upload various file types, processes them into manageable chunks, embeds these chunks, and stores them in a vector database. When a user asks a question, the system retrieves the most relevant chunks from the stored data and uses a Large Language Model (LLM) — specifically **SEALION V 3.5 8B** — to generate a coherent and contextually accurate answer. It also provides transparency by displaying the top retrieved chunks that informed the answer.

## Features

* **File Uploads:** Supports uploading various document types for processing.
* **Question Answering:** Allows users to ask natural language questions about the uploaded content.
* **Contextual Answers:** Generates answers grounded in the content of the uploaded documents.
* **Top Chunk Display:** Shows the top 3 most relevant retrieved chunks to provide transparency and allow users to verify the source of the answer.
* **Vector Database Integration:** Utilizes ChromaDB for efficient storage and retrieval of document embeddings.
* **Modular Design:** Separates core RAG logic (`SEALION_RAG.py`) from the application interface (`app.py`).
* **Local LLM Support:** Uses **SEALION V 3.5 8B**, an open-weight LLM, for offline and customizable deployment.

## Technologies Used

* **Python:** The core programming language for the application logic.
* **ChromaDB:** An open-source embedding database for storing and querying vector embeddings.
* **LangChain (Inferred):** Likely used for orchestrating the RAG pipeline (chunking, embedding, retrieval, LLM integration).
* **Streamlit (Inferred from `app.py`):** A framework for building the user interface.
* **Large Language Model (LLM):** **SEALION V 3.5 8B** is used to generate context-aware answers.
* **Hugging Face Transformers (Inferred):** For managing embeddings and integrating local LLMs.


## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.11+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/NDarayut/RAG-Prototype.git](https://github.com/NDarayut/RAG-Prototype.git)
    cd RAG-Prototype
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure your virtual environment is activated.**
2.  **Run the main application:**
    ```bash
    python app.py
    ```
3.  **Open your web browser** and navigate to the address provided in your terminal (e.g., `http://localhost:8501` if using Streamlit).
4.  **Upload your documents** using the provided interface.
5.  **Enter your question** in the input field and submit.
6.  The application will display the generated answer along with the top 3 retrieved chunks that were used to formulate the response.

## Project Structure  
```
RAG-Prototype/
├── chroma/             # Directory for ChromaDB persistent storage
├── data/               # Placeholder for uploaded or processed data
├── app.py              # Main application file (likely Streamlit/Flask/FastAPI UI)
├── SEALION_RAG.py              # Core RAG logic (chunking, embedding, retrieval, generation)
├── requirements.txt    # List of Python dependencies
└── .gitignore          # Specifies intentionally untracked files to ignore
```

## License

Distributed under the MIT License. See `LICENSE` for more information. (Note: A `LICENSE` file is assumed to be present or will be added. If not, please create one in your repository.)

## Contact

NDarayut - [Gmail](darayutnhem009@gmail.com)

Project Link: [https://github.com/NDarayut/RAG-Prototype](https://github.com/NDarayut/RAG-Prototype)