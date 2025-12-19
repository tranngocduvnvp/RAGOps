from pydantic.v1 import BaseModel, Field, validator
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbedding(BaseEmbedding, Embeddings):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(self.config.name, trust_remote_code=True, device="cpu", cache_folder="./checkpoint_model")

    def encode(self, text: str):
        return self.embedding_model.encode(text)
    # Các method bắt buộc của LangChain Embeddings
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed một list các documents."""
        return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed một query đơn."""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()