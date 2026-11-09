from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.retrieval.rag import answer_question
import logging
from fastapi import FastAPI, HTTPException

logger = logging.getLogger("uvicorn.error")

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
    try:
        answer = await answer_question(body.question)
    except Exception:
        logger.exception("RAG failed")
        raise HTTPException(status_code=503, detail="Answer service is temporarily unavailable. Please try again.")
    return AskOut(answer=answer)