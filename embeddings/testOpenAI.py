from langchain.embeddings.base import Embeddings
import requests
from typing import List
from typing import List, Dict, Any

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.url = f"{base_url}/embeddings" 

    def encode(self, text: str):
        response = requests.post(self.url, json={"input": text, "normalize_embeddings": False})
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(self.url, json={"input": texts, "normalize_embeddings": True})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(self.url, json={"input": text, "normalize_embeddings": True})
        response.raise_for_status()
        return response.json()["embeddings"][0]
    


embeddings = CustomSentenceTransformerEmbeddings()
print(embeddings.embed_query("Xin chào"))
print(embeddings.embed_documents(["Text 1", "Text 2"]))

