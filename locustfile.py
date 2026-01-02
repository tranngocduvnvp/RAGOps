from locust import HttpUser, task, between
import json
import random


SAMPLE_TEXTS = [
    "Đây là một câu văn tiếng Việt để kiểm tra embedding.",
    "Machine learning là một lĩnh vực rất thú vị.",
    "FastAPI là framework web nhanh và hiện đại cho Python.",
    "Hôm nay thời tiết rất đẹp, trời trong xanh.",
    "Embedding model giúp biểu diễn văn bản dưới dạng vector.",
    "Sentence Transformers là thư viện mạnh mẽ cho semantic search.",
    "Colab là môi trường tuyệt vời để thử nghiệm mô hình AI."
]

class EmbeddingUser(HttpUser):
    wait_time = between(0.1, 0.5)  

    @task
    def embed_single_text(self):
        # Test with one sentence
        payload = {
            "requests": [{
            "input": random.choice(SAMPLE_TEXTS),
            "normalize_embeddings": True
            }]
        }

        # payload = {
        #     "input": random.choice(SAMPLE_TEXTS),
        #     "normalize_embeddings": True
        # }

        self.client.post("/embeddings", json=payload)

    # @task(3)  # Test with multiple sentences
    # def embed_batch(self):
    #     batch_size = random.randint(2, 10)
    #     texts = random.choices(SAMPLE_TEXTS, k=batch_size)
    #     # payload = {
    #     #     "input": texts,
    #     #     "normalize_embeddings": True
    #     # }

    #     payload = {
    #         "requests": [{
    #         "input": texts,
    #         "normalize_embeddings": True
    #         }]
    #     }

    #     self.client.post("/embeddings", json=payload)