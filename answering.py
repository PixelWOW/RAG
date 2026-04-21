import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


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

query = "When was tesla Roadster launched?"

retriever = db.as_retriever(search_kwargs= {"k":5})

relevant_docs = retriever.invoke(query)

#answering query
# print("\nAnswering user query...\n")
# for i, docs in enumerate(relevant_docs,1):
#     print(f"Document {i}:\n{docs.page_content}\n")

combined_input = f"""Based on the following documents, answer the following question: {query}
Documents: {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using information only from these documents.If you can't find the answer in these documents, clearly say "I don't have enough information about the query based on the provided documents."
"""

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-4-31b-it",
#     huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_KEY"),
#     temperature = 0.2,
#     max_new_tokens=1024
#     # task = "text-generation"
# )
# chat_model= ChatHuggingFace(llm= llm)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0.2
)

message = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = combined_input)
]

# llm = Together(
#     model="mistralai/Mistral-7B-Instruct-v0.1",
#     temperature=0.2
# )

print("\nGenerating response\n")
result = llm.invoke(message)
print(result.content)