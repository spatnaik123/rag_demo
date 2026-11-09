from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name="rag-demo-vectors", embedding=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def answer_question(question: str) -> str:
    results = await vector_store.asimilarity_search_with_score(question, k=5)
    #print([round(score, 2) for _, score in results])   # temporary, to tune the threshold
    relevant = [doc for doc, score in results if score >= 0.4]
    if not relevant:
        return "I don't know."
    context = "\n\n".join(doc.page_content for doc in relevant)

    prompt = f"""
                Answer the question using only the information provided in the context.
                If the answer is not available in the context, say "I don't know."

                Context:
                {context}

                Question:
                {question}
                """
    response = await llm.ainvoke(prompt)
    return response.content