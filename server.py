import sys
print(f"🟢 Python running: {sys.executable}")
import torch
print(torch.cuda.is_available())
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import logging
from app.schemas import Query, Response
from app.inference import generate_response
from app.ocr_service import router as ocr_router

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.include_router(ocr_router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=Response)
async def ask_model(query: Query):
    logging.info(f"Received prompt: {query.prompt}")
    result = generate_response(query.prompt)
    return Response(response=result)

class Feedback(BaseModel):
    user_id: int
    prompt: str
    response: str
    label: str  # например, "bad"

@app.post("/feedback")
async def receive_feedback(feedback: Feedback):
    feedback_path = Path("data/bad_feedback.jsonl")
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "a", encoding="utf-8") as f:
        json.dump(feedback.dict(), f, ensure_ascii=False)
        f.write("\n")
    logging.info(f"📥 Получен фидбек от пользователя {feedback.user_id}")
    return {"status": "ok"}

# python -m uvicorn server:app --reload --port 8000

# conda activate .\.conda
