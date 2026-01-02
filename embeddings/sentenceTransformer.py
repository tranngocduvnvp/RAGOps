from pydantic.v1 import BaseModel, Field, validator
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain.embeddings.base import Embeddings
import requests
import torch

class SentenceTransformerEmbedding(BaseEmbedding, Embeddings):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(self.config.name, trust_remote_code=True, device="cpu", cache_folder="./checkpoint_model")

    def encode(self, text: str):
        return self.embedding_model.encode(text)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
    

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.url = f"{base_url}/embeddings"  

    def encode(self, text: str):
        response = requests.post(self.url, json={"requests":[{"input": text, "normalize_embeddings": False}]})
        response.raise_for_status()
        return torch.tensor(response.json()[0]["embeddings"][0])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(self.url, json={"requests":[{"input": texts, "normalize_embeddings": True}]})
        response.raise_for_status()
        return response.json()[0]["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(self.url, json={"requests":[{"input": text, "normalize_embeddings": True}]})
        response.raise_for_status()
        return response.json()[0]["embeddings"][0]