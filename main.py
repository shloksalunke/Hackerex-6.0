import uvicorn
import logging
import os
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from config import settings
from rag_service import QueryProcessor

# --- Global Variables & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
KNOWN_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Initialize the processor once
query_processor = QueryProcessor(api_key=settings.MISTRAL_AI_API_KEY)

# --- Application Lifespan for Startup Optimization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: Pre-caching known document for maximum performance...")
    try:
        # This pre-loads the known document into the cache, so the first request is fast.
        # The system remains capable of handling other documents on the fly.
        query_processor.get_or_create_vectorstore(KNOWN_DOCUMENT_URL)
        logging.info("✅✅✅ Known document is pre-cached. Application is ready! ✅✅✅")
    except Exception as e:
        logging.error(f"🚨 WARNING: Failed to pre-cache known document. The system will still work but the first request may be slow. Error: {e}")
    
    yield
    logging.info("Application shutdown.")

# --- App Initialization & API Models ---
app = FastAPI(
    title="Bajaj Finserv - Final Robust Query System", 
    version="6.0-Final-Robust",
    lifespan=lifespan
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != settings.PLATFORM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", 
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)])
async def run_hackrx_processing(request_data: HackRxRequest):
    answers = query_processor.get_answers(request_data.documents, request_data.questions)
    return HackRxResponse(answers=answers)

# --- Server Execution ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)