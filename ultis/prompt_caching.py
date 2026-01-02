import redis
import hashlib
import json
import numpy as np
from typing import Optional, Tuple, List, Any
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbedding(BaseEmbedding, Embeddings):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(
            self.config.name,
            trust_remote_code=True,
            device="cpu",
            cache_folder="./checkpoint_model"
        )

    def encode(self, text: str):
        return self.embedding_model.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

class PromptCache:
    def __init__(
        self,
        embedding_instance,
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        expire_seconds=3600,
        similarity_threshold=0.85,
        index_name='rag_prompt_idx',
        vector_prefix='rag_vector:',
        password=""
    ):
        self.embedding_model = embedding_instance
        self.vector_dim = len(self.embedding_model.embed_query("test"))

        self.redis_client = redis.StrictRedis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True, password=password
        )
        self.expire_seconds = expire_seconds
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
        self.vector_prefix = vector_prefix

        self._create_vector_index()

    def _get_hash_key(self, user_id: str, prompt: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        return f"rag_prompt:{user_id}:{prompt_hash}"

    def _get_vector_key(self, user_id: str, prompt_hash: str) -> str:
        return f"{self.vector_prefix}{user_id}:{prompt_hash}"

    def _create_vector_index(self):
        try:
            self.redis_client.ft(self.index_name).info()
        except:
            from redis.commands.search.field import VectorField

            schema = (
                VectorField("vector", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE"
                }),
            )
            self.redis_client.ft(self.index_name).create_index(schema)

    def _embed_prompt(self, prompt: str) -> np.ndarray:
        embedding_list = self.embedding_model.embed_query(prompt)
        return np.array(embedding_list, dtype=np.float32)

    def get_cached_response(
        self,
        user_id: str,
        prompt: str,
        use_vector_search: bool = True
    ) -> Tuple[Optional[Any], bool, Optional[float]]:
        exact_key = self._get_hash_key(user_id, prompt)
        cached = self.redis_client.get(exact_key)
        if cached:
            response = json.loads(cached) if cached.startswith(('{', '[')) else cached
            return response, True, 1.0

        if use_vector_search:
            try:
                from redis.commands.search.query import Query

                query_vector = self._embed_prompt(prompt)
                query_blob = query_vector.tobytes()

                q = (
                    Query("*=>[KNN 1 @vector $query AS score]")
                )

                res = self.redis_client.ft(self.index_name).search(
                    q,
                    query_params={"query": query_blob}
                )

                if res.total > 0:
                    doc = res.docs[0]
                    distance = float(doc.score)
                    similarity = 1 - distance

                    if similarity >= self.similarity_threshold:
                        similar_hash = doc.id.split(":")[-1]
                        similar_key = f"rag_prompt:{user_id}:{similar_hash}"
                        cached_similar = self.redis_client.get(similar_key)
                        if cached_similar:
                            response = json.loads(cached_similar) if cached_similar.startswith(('{', '[')) else cached_similar
                            return response, False, round(similarity, 4)

            except Exception as e:
                print(f"Vector search error: {e}")

        return None, False, None

    def cache_response(self, user_id: str, prompt: str, response: Any):
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        exact_key = self._get_hash_key(user_id, prompt)
        vector_key = self._get_vector_key(user_id, prompt_hash)

        serialized = json.dumps(response) if isinstance(response, (dict, list)) else str(response)

        self.redis_client.set(exact_key, serialized, ex=self.expire_seconds)

        embedding = self._embed_prompt(prompt)
        vector_blob = embedding.tobytes()
        self.redis_client.hset(vector_key, mapping={"vector": vector_blob})
        self.redis_client.expire(vector_key, self.expire_seconds)

    def clear_user_cache(self, user_id: str):
        exact_keys = self.redis_client.keys(f"rag_prompt:{user_id}:*")
        vector_keys = self.redis_client.keys(f"{self.vector_prefix}{user_id}:*")
        all_keys = exact_keys + vector_keys
        if all_keys:
            self.redis_client.delete(*all_keys)
        print(f"User {user_id}'s cache has been cleared.")

    def clear_all_cache(self):
        exact_keys = self.redis_client.keys("rag_prompt:*")
        vector_keys = self.redis_client.keys(f"{self.vector_prefix}*")
        all_keys = exact_keys + vector_keys
        if all_keys:
            self.redis_client.delete(*all_keys)
        print("All cache has been cleared.")
    

