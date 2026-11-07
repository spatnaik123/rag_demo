import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.ingestion.load_and_chunk import load_and_chunk_documents


# PART 1: Load API keys
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("API keys are missing.")


# PART 2: Create OpenAI embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# PART 3: Load chunks from Step 4
chunks = load_and_chunk_documents()

print(f"Total chunks received: {len(chunks)}")


# PART 4: Connect to Pinecone and store chunks
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="rag-demo-vectors"
)

print("All chunks stored successfully in Pinecone!")