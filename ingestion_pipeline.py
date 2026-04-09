import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain-huggingface import HuggingfaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_docs(docspath = "docs"):
    
    if not os.path.exists(docspath):
        raise FileNotFoundError(f"No directory found named {docspath}.")

    loader = DirectoryLoader(
        path = docspath,
        glob = "*.txt",
        loader_cls = TextLoader, #text loader class
        loader_kwargs={'encoding': 'utf-8'}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No files in {docspath} directory. Please add the files.")
    
    for i, doc in enumerate(documents[:2]): #first 2 docs
        print(f"Document {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...") #first 99 chars
        print(f"Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800,chunk_overlap=0):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    for i,chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1}:")
        print(f"Preview: {chunk.page_content[:100]}")
        print(f"Chunk length: {len(chunk.page_content)}\n")
    
    return chunks

def vector_store(chunks, persist_dir = "db/chroma.db"):
    
    embedding_model = HuggingFaceEndpointEmbeddings(
        model = "BAAI/bge-base-en-v1.5",
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_KEY")
    )
    print("---Creating vector data store---")
    
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_dir,
        collection_metadata = {"hnsw:space":"cosine"}
    )

    print(f"Finished creating vector score\nVector store present in {persist_dir}")
    return vectorstore



def main():
    print("Hello main")

    # 1. Load files
    documents = load_docs(docspath = "docs")
    # 2. Chunk files
    chunks = split_documents(documents)
    # 3. Embbed into vector
    vector = vector_store(chunks)

if __name__ == "__main__":
    main()