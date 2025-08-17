# RAG Chatbot

This project demonstrates a simple Retrieval-Augmented Generation (RAG) system using AWS Bedrock for embeddings and ChromaDB as a local vector store.

## Files

- `db_populate.py`: Reads PDF documents from the `data/` folder, creates embeddings with AWS Bedrock, and stores them in ChromaDB.

- `get_embedding_function.py`: A helper file that configures and returns the Bedrock embedding client.

- `query_data.py`: Loads the embeddings from ChromaDB, takes a query, and finds the most relevant document chunks to inform an answer.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- AWS CLI configured with Bedrock access
- Install dependencies: `pip install langchain langchain-chroma boto3 pypdf python-dotenv`

### 2. Populate the Database

Place your PDF files in a new `data/` folder, then run:

```bash
python db_populate.py
```
### 3. Query Your Documents
Run the script to ask questions about your data:

```bash
python query_data.py
