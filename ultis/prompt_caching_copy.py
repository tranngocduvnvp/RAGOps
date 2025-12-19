import redis
import hashlib
import json
import numpy as np
from typing import Optional, Tuple, List, Any

# Import class embedding của bạn
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbedding(BaseEmbedding, Embeddings):
    # Class của bạn giữ nguyên như vậy
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
        embedding_instance: SentenceTransformerEmbedding,
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        expire_seconds=3600,
        similarity_threshold=0.85,
        index_name='rag_prompt_idx',
        vector_prefix='rag_vector:'  # Có thể giữ hoặc tùy chỉnh
    ):
        self.embedding_model = embedding_instance
        self.vector_dim = len(self.embedding_model.embed_query("test"))

        self.redis_client = redis.StrictRedis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        self.expire_seconds = expire_seconds
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
        self.vector_prefix = vector_prefix

        self._create_vector_index()

    def _get_hash_key(self, user_id: str, prompt: str) -> str:
        """Key exact match: bao gồm cả user_id"""
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        return f"rag_prompt:{user_id}:{prompt_hash}"

    def _get_vector_key(self, user_id: str, prompt_hash: str) -> str:
        """Key vector: cũng bao gồm user_id"""
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
            print(f"Tạo vector index '{self.index_name}' thành công.")

    def _embed_prompt(self, prompt: str) -> np.ndarray:
        embedding_list = self.embedding_model.embed_query(prompt)
        return np.array(embedding_list, dtype=np.float32)

    # === Cập nhật các method để nhận user_id ===
    def get_cached_response(
        self,
        user_id: str,
        prompt: str,
        use_vector_search: bool = True
    ) -> Tuple[Optional[Any], bool, Optional[float]]:
        # 1. Exact match (giữ nguyên - hoạt động tốt)
        exact_key = self._get_hash_key(user_id, prompt)
        cached = self.redis_client.get(exact_key)
        if cached:
            response = json.loads(cached) if cached.startswith(('{', '[')) else cached
            return response, True, 1.0

        # 2. Vector similarity search - PHIÊN BẢN SỬA CHỮA ỔN ĐỊNH
        if use_vector_search:
            try:
                from redis.commands.search.query import Query

                query_vector = self._embed_prompt(prompt)
                query_blob = query_vector.tobytes()

                user_vector_prefix = f"{self.vector_prefix}{user_id}:"

                # Build query đúng cách: filter prefix + KNN
                q = (
                    Query("*=>[KNN 1 @vector $query AS score]")
                    # .filter(f"@__key:{user_vector_prefix}*")  # Filter theo prefix key
                    # .dialect(2)  # Bắt buộc dùng dialect 2 để hỗ trợ __key
                    # .return_fields("score")
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
                # Fallback: không dùng vector nếu lỗi

        return None, False, None

    def cache_response(self, user_id: str, prompt: str, response: Any):
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        exact_key = self._get_hash_key(user_id, prompt)
        vector_key = self._get_vector_key(user_id, prompt_hash)

        serialized = json.dumps(response) if isinstance(response, (dict, list)) else str(response)

        # Lưu exact response
        self.redis_client.set(exact_key, serialized, ex=self.expire_seconds)

        # Lưu vector
        embedding = self._embed_prompt(prompt)
        vector_blob = embedding.tobytes()
        self.redis_client.hset(vector_key, mapping={"vector": vector_blob})
        self.redis_client.expire(vector_key, self.expire_seconds)

    def clear_user_cache(self, user_id: str):
        """Xóa toàn bộ cache của một user cụ thể"""
        exact_keys = self.redis_client.keys(f"rag_prompt:{user_id}:*")
        vector_keys = self.redis_client.keys(f"{self.vector_prefix}{user_id}:*")
        all_keys = exact_keys + vector_keys
        if all_keys:
            self.redis_client.delete(*all_keys)
        print(f"Đã xóa cache của user {user_id}")

    def clear_all_cache(self):
        """Xóa toàn bộ cache (tất cả user)"""
        exact_keys = self.redis_client.keys("rag_prompt:*")
        vector_keys = self.redis_client.keys(f"{self.vector_prefix}*")
        all_keys = exact_keys + vector_keys
        if all_keys:
            self.redis_client.delete(*all_keys)
        print("Đã xóa toàn bộ cache.")
    

# ==================== CÁCH SỬ DỤNG ====================

# 1. Tạo config và embedding instance (theo hệ thống của bạn)
embedding_config = EmbeddingConfig(name="Alibaba-NLP/gte-multilingual-base")  # hoặc bất kỳ model nào bạn dùng
my_embedding = SentenceTransformerEmbedding(config=embedding_config)

# 2. Khởi tạo cache với instance embedding
cache = PromptCache(
    embedding_instance=my_embedding,
    expire_seconds=86400,
    similarity_threshold=0.82
)

# 3. Hàm query với cache
def fake_llm(prompt):
        return f"LLM answer for: '{prompt}'"

def query_with_cache(user, prompt):
    response, is_exact, sim = cache.get_cached_response(user, prompt)
    if response:
        source = "EXACT cache" if is_exact else f"SIMILAR cache (sim={sim:.3f})"
        print(f"[HIT] {source}: {response}")
    else:
        print("[MISS] Gọi LLM...")
        response = fake_llm(prompt)
        cache.cache_response(user, prompt, response)
        print(f"[STORED] {response}")
    return response

user = "2345"
# Test
query_with_cache(user, "What is Retrieval-Augmented Generation?")
query_with_cache(user, "What is Retrieval-Augmented Generation?")  # exact hit
query_with_cache(user, "Can you explain what RAG is in AI?")       # similar hit
query_with_cache(user, "Tell me about transformers architecture")  # miss
query_with_cache(user, "What is transformer achit")
# cache.clear_all_cache()