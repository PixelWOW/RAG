import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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


def main():
    print("Hello main")

    # 1. Load files
    documents = load_docs(docspath = "docs")
    # 2. Chunk files
    # 3. Embbed into vector

if __name__ == "__main__":
    main()