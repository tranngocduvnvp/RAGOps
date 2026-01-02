from __future__ import annotations

import os
import typing as t

import numpy as np
import bentoml
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from logging_config import setup_console_logging

logger_app = setup_console_logging(app_name="BentoML Embedding", log_level="INFO") 


# ====================== CONFIG ======================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_FOLDER = "./checkpoint_model"  
DEVICE =  "cpu"


class EmbedRequest(BaseModel): 
    input: t.Union[str, t.List[str]] = Field(..., description="Text or list text")
    normalize_embeddings: bool = Field(True, description="Normalize embeddings or not")

class EmbedResponse(BaseModel):
    embeddings: t.List[t.List[float]]
    model: str
    usage: dict

# ====================== BENTO SERVICE ======================
@bentoml.service(
    name="embedding_service",
    traffic={"timeout": 60},  
    resources={"cpu": 2},     
)
class EmbeddingService:
    def __init__(self) -> None:
        logger_app.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=DEVICE,
            cache_folder=CACHE_FOLDER if CACHE_FOLDER else None,
        )
        logger_app.info(f"Embedding model loaded on {DEVICE}")

    @bentoml.api(
        batchable=True,                          
        max_batch_size=8,                        
        max_latency_ms=1000,                      
    )
    def embeddings(
        self,
        requests: t.List[EmbedRequest],
    ) -> t.List[EmbedResponse]:
        
   
        all_texts: t.List[str] = []
        all_normalize: t.List[bool] = []
        indices: t.List[int] = []  

        # print("len request:", len(requests))
        for idx, req in enumerate(requests):
            # texts = [req["input"]] if isinstance(req["input"], str) else req["input"]
            texts = [req.input] if isinstance(req.input, str) else req.input
            if not texts:
                
                raise ValueError("Input texts cannot be empty")

            all_texts.extend(texts)
            all_normalize.extend([req.normalize_embeddings] * len(texts))
            indices.extend([idx] * len(texts))

        
        embeddings_np: np.ndarray = self.model.encode(
            all_texts,
            normalize_embeddings=all_normalize[0],  
            batch_size=len(all_texts),             
            show_progress_bar=False,
        )

        embeddings_list = embeddings_np.tolist()

       
        responses = []
        pos = 0
        for req_idx, req in enumerate(requests):
            texts = [req.input] if isinstance(req.input, str) else req.input
            batch_emb = embeddings_list[pos : pos + len(texts)]
            pos += len(texts)

            responses.append(
                EmbedResponse(
                    embeddings=batch_emb,
                    model=EMBEDDING_MODEL_NAME,
                    usage= {"prompt_tokens": len(texts), "total_tokens": len(texts)}
                )
            )

        return responses

    @bentoml.api(route="/")
    def health(self) -> dict:
        return {
            "message": "Embedding API is running!",
            "device": DEVICE,
            "model": EMBEDDING_MODEL_NAME,
        }