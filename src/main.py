import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.retrieval.rag import answer_question, stream_answer

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


@app.post("/ask/stream")
async def ask_stream(body: AskIn):
    async def event_generator():
        try:
            async for piece in stream_answer(body.question):
                yield f"data: {json.dumps(piece)}\n\n"
        except Exception:
            logger.exception("Streaming failed")
            yield "event: error\ndata: Answer service is temporarily unavailable.\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")