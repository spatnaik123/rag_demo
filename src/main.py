from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

@app.get("/health")
def health():
    return {'status': 'ok'}

class AskIn(BaseModel):
    question: str = Field(min_length=3)

class AskOut(BaseModel):
    answer: str

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    return AskOut(answer=f"You asked: {body.question}")