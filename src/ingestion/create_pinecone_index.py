import os

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

# Load API Key
load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create Index
pc.create_index(name="rag-demo-vectors",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("Vector index created successfully!")