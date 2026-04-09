## RAG Document Ingestion Pipeline

This project implements the **Ingestion** phase of a Retrieval-Augmented Generation (RAG) system. It loads text documents, splits them into manageable chunks using recursive character splitting, and stores them in a **ChromaDB** vector store using **Hugging Face** embeddings.

---

### Project Structure

* **`docs/`**: Folder containing your raw `.txt` source files (e.g., Google.txt, Microsoft.txt).
* **`db/`**: The directory where the persisted ChromaDB vector store is saved.
* **`ingestion_pipeline.py`**: The main Python script for processing documents.
* **`.env`**: Stores sensitive API keys.

---

### Prerequisites

1.  **Python 3.10 - 3.12** (Recommended).
2.  **Hugging Face Account**: To get a free Inference API token.

#### Installation
```bash
pip install langchain-community langchain-huggingface langchain-chroma python-dotenv
```

---

### Setup

1.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    HUGGINGFACEHUB_API_KEY=your_huggingface_api_token_here
    ```

2.  **Prepare Documents**:
    Place your `.txt` files inside a folder named `docs`.

---

### How It Works

1.  **Loading**: Uses `DirectoryLoader` and `TextLoader` to read all text files from the `docs/` directory.
2.  **Splitting**: Uses `RecursiveCharacterTextSplitter` to break documents into chunks (default: **800** characters with **80** character overlap). This ensures semantic context is preserved across chunks.
3.  **Embedding**: Chunks are converted into numerical vectors using the `BAAI/bge-base-en-v1.5` model via the Hugging Face Inference API.
4.  **Vector Storage**: The vectors are stored in **ChromaDB** using **Cosine Similarity** (`hnsw:space: 'cosine'`) to measure document relevance.



---

### Usage

Run the pipeline from your terminal:
```bash
python ingestion_pipeline.py
```

Upon success, you will see a `db/` folder created in your project directory containing the indexed data.

---

### Troubleshooting

* **Import Errors**: Ensure you have installed `langchain-huggingface`. The script uses `from langchain_huggingface import HuggingFaceEndpointEmbeddings`.
* **API Issues**: If you get a validation error, ensure the parameter used is `huggingfacehub_api_token` and that your token is valid.
* **Vector Mismatch**: If you change embedding models, **delete the `db/` folder** before re-running the script to prevent dimension conflicts.
