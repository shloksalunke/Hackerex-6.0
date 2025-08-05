# --- START OF FILE main.py ---
import uvicorn
import logging
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from config import settings
from rag_service import QueryProcessor

# --- Global Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
KNOWN_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Initialize the core query processing engine
query_processor = QueryProcessor(api_key=settings.MISTRAL_AI_API_KEY)

# --- Application Lifespan for pre-warming the cache ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: Initiating cache warm-up for the known document...")
    try:
        # This will download, process, and cache the vector store for the known document
        # so that the first API call is fast.
        query_processor.get_vectorstore(KNOWN_DOCUMENT_URL)
        logging.info("✅✅✅ Cache is warm. Application is ready to accept requests! ✅✅✅")
    except Exception as e:
        logging.error(f"🚨 CRITICAL WARNING: Failed to warm up cache for known document. The first request might be slow. Error: {e}")
    
    yield
    logging.info("Application shutdown.")

# --- App Initialization & API Models ---
app = FastAPI(
    title="Bajaj Finserv - Elite Performance RAG System", 
    version="11.0-Elite-Advanced",
    lifespan=lifespan
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# --- CRITICAL: This response model correctly matches the platform's expected output format ---
class HackRxResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
auth_scheme = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Validates the Bearer token against the one in the environment."""
    if credentials.scheme != "Bearer" or credentials.credentials != settings.PLATFORM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", 
          response_model=HackRxResponse,
          dependencies=[Depends(verify_token)]) # Enforces authentication for this endpoint
async def run_hackrx_processing(request_data: HackRxRequest):
    """
    Receives a document URL and a list of questions, returns a list of answers.
    """
    logging.info(f"Received request for document: {request_data.documents} with {len(request_data.questions)} questions.")
    answers = query_processor.get_answers(request_data.documents, request_data.questions)
    logging.info("Successfully processed request and generated answers.")
    return HackRxResponse(answers=answers)

# --- Server Execution ---
if __name__ == "__main__":
    # Use the PORT environment variable if available (for deployment platforms like Heroku)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
# --- END OF FILE main.py ---