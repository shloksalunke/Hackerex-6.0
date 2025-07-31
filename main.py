import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

from config import settings
from rag_service import QueryProcessor

# --- Global Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- App Initialization (Stateless for Vercel) ---
app = FastAPI(
    title="Bajaj Finserv - Vercel Deployed Query System",
    version="7.0-Vercel"
)

# --- Authentication ---
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != settings.PLATFORM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run",
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)])
async def run_hackrx_processing(request_data: HackRxRequest):
    """
    This endpoint is designed for serverless environments. It creates a new
    processor on each request, which is less performant but Vercel-compatible.
    """
    try:
        processor = QueryProcessor(api_key=settings.MISTRAL_AI_API_KEY)
        answers = processor.get_answers(request_data.documents, request_data.questions)
        return HackRxResponse(answers=answers)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# This block is for local testing only; Vercel does not use it.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)