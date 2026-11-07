import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

# Load API Key from .env file
load_dotenv()

# Step 1.1: Convert user query to Embedding

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Pinecone database

vector_store = PineconeVectorStore(index_name='rag-demo-vectors',
                                         embedding=embeddings)

# STEP 1.2: Temporary test query
query = input("Enter your question: ")

# Step 2 : Search Pinecone using similarity (cosine similarity)
results = vector_store.similarity_search(query, k=5)

print(f"Retrieved {len(results)} relevant chunks.")


for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(result.page_content)
    print("Source:", result.metadata.get("source"))


################################################################

#####################  GOAL  ##################

### Combine the user's query + the 10 retrieved chunks into one context that will later be sent to the LLM.

### In short below code is to prepare the Augmented Prompt = User Query + Retrived 5 chunk

### The augmented_prompt is what we will send to the LLM

# STEP 3: Augmentation
# Combine the user query with the retrieved chunks

context = "\n\n".join(
    result.page_content for result in results
)

augmented_prompt = f"""
Answer the question using only the information provided in the context.
If the answer is not available in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""

print("\n===== AUGMENTED CONTEXT =====")
print(augmented_prompt)

#####        Goal: Send the augmented prompt to an LLM and get the final answer.

# STEP 4: Send augmented prompt to the LLM

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke(augmented_prompt)

print("\n===== FINAL ANSWER =====")
print(response.content)