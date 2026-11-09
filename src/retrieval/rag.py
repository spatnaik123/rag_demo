from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name="rag-demo-vectors", embedding=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

NO_ANSWER = "I don't know."


async def build_prompt(question: str):
    results = await vector_store.asimilarity_search_with_score(question, k=5)
    relevant = [doc for doc, score in results if score >= 0.4]
    if not relevant:
        return None
    context = "\n\n".join(doc.page_content for doc in relevant)
    return f"""
Answer the question using only the information provided in the context.
If the answer is not available in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""


async def answer_question(question: str) -> str:
    prompt = await build_prompt(question)
    if prompt is None:
        return NO_ANSWER
    response = await llm.ainvoke(prompt)
    return response.content


async def stream_answer(question: str):
    prompt = await build_prompt(question)
    if prompt is None:
        yield NO_ANSWER
        return
    async for chunk in llm.astream(prompt):
        yield chunk.content