
from sentence_transformers import CrossEncoder
import numpy as np
import requests


class Reranker():
    def __init__(self, base_url: str = "http://localhost:7777"):
        self.url = f"{base_url}/rerank"  
        self.reranker_name = "Access to rerank service to check model name"

    def __call__(self, query: str, passages: list):
        # Combine query and passages into pairs
        query_passage_pairs = [[query, passage["combined_information"]] for passage in passages]

        # Get scores from the reranker model
        scores = requests.post(self.url, json={"requests":[{"data":query_passage_pairs}]}).json()[0]

        # Sort passages based on scores
        ranked_passages = [passage for _, passage in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
        ranked_scores = sorted(scores, reverse=True)
        
        # Convert scores to standard Python floats
        ranked_scores = [float(score) for score in ranked_scores]
        # Return just the passages in ranked order
        return ranked_scores, ranked_passages
