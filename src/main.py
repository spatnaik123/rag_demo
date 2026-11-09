from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.retrieval.rag import answer_question

app = FastAPI()

@app.get("/health")
def health():
    return {'status': 'ok'}

class AskIn(BaseModel):
    question: str = Field(min_length=3)

class AskOut(BaseModel):
    answer: str

@app.post("/ask", response_model=AskOut)
async def ask(body: AskIn):
    answer = await answer_question(body.question)
    return AskOut(answer=answer)