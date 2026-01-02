from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import uvicorn
from contextlib import asynccontextmanager
from logging_config import setup_console_logging

logger_app = setup_console_logging(app_name="FastAPI Embedding", log_level="INFO") 

embedding_model: SentenceTransformer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    logger_app.info("Starting Embedding FASTAPIServer ....")
    embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=DEVICE,
            cache_folder=CACHE_FOLDER
        )
    logger_app.info("Load FASTAPIServer successfully!")
    yield
    logger_app.info("Shutdown Complete")


app = FastAPI(
    title="Embedding API",
    description="Embedding FASTAPIServer",
    version="1.0",
    lifespan=lifespan
)

# ====================== CONFIG ======================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_FOLDER = "./checkpoint_model"
DEVICE = "cpu" 


# ====================== EMBEDDING ======================
class EmbedRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text hoặc list texts cần embed")
    normalize_embeddings: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    usage: dict


@app.post("/embeddings", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input texts cannot be empty")

    try:
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=request.normalize_embeddings
        ).tolist()

        return EmbedResponse(
            embeddings=embeddings,
            model=EMBEDDING_MODEL_NAME,
            usage={"prompt_tokens": len(texts), "total_tokens": len(texts)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")




# ====================== HEALTH CHECK ======================
@app.get("/")
async def root():
    return {
        "message": "Embedding API is running!",
        "device": DEVICE
    }


if __name__ == "__main__": 
    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)
