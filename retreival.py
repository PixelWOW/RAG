import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_dir = "db/chroma.db"

embedding_model = HuggingFaceEndpointEmbeddings(
        model = "BAAI/bge-base-en-v1.5",
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_KEY")
)

db = Chroma(
    embedding_function = embedding_model,
    persist_directory = persistent_dir,
    collection_metadata = {"hnsw:space":'cosine'}
)

query = input("Enter your query: ")

retriever = db.as_retriever(search_kwargs= {"k":3})

relevant_docs = retriever.invoke(query)

print("\nAnswering user query...\n")
for i, docs in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{docs.page_content}\n")