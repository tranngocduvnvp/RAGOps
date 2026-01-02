from __future__ import annotations
import bentoml
from typing import List
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from logging_config import setup_console_logging
import os

logger_app = setup_console_logging(app_name="BentoML Reranking", log_level="INFO") 

# ====================== CONFIG ======================
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2")
CACHE_FOLDER = "./checkpoint_model"  
DEVICE =  "cpu"

class RerankRequest(BaseModel):
    data: List[List[str]]  # List of [query, document] pairs


@bentoml.service(
    resources={"cpu": 1},  
    traffic={"timeout": 60},
)
class RerankService:
    def __init__(self) -> None:
        logger_app.info(f"Loading rerank model: {RERANK_MODEL_NAME} on {DEVICE}")
        self.reranker = CrossEncoder(
            RERANK_MODEL_NAME,
            trust_remote_code=True,
            device=DEVICE, 
            cache_folder=CACHE_FOLDER,
        )
        logger_app.info(f"Rerank model loaded on {DEVICE}")

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=2, max_latency_ms=1000)  # Kích hoạt adaptive batching
    def rerank(self, requests: List[RerankRequest]) -> List[List[float]]:
        all_pairs: List[List[str]] = []
        for req in requests:
            all_pairs.extend(req.data)

        scores = self.reranker.predict(all_pairs)

        scores_list = scores.tolist()

        result: List[List[float]] = []
        idx = 0
        for req in requests:
            num_pairs = len(req.data)
            result.append(scores_list[idx : idx + num_pairs])
            idx += num_pairs

        return result