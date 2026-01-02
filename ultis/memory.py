import os
from typing import List, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer
from re_rank import Reranker
from embeddings import BaseEmbedding, EmbeddingConfig, SentenceTransformerEmbedding, CustomSentenceTransformerEmbeddings


class MemoryModule:
    """
    The memory module supports multi-session for RAG, using local SentenceTransformer for embedding.
    - Short Term: Separate ConversationBufferMemory per session.
    - Long Term: Separate FAISS per session, stored on disk.
    """

    def __init__(
        self,
        embedding_model,
        reranker_model,
        faiss_base_path: str = "faiss_indexes",
        max_short_term_messages: int = 2,
    ):
        
        self.embeddings = embedding_model
        self.reranker_model = reranker_model

        self.faiss_base_path = faiss_base_path
        self.max_short_term_messages = max_short_term_messages

        self.short_term_memories: Dict[str, ConversationBufferMemory] = {}
        self.long_term_memories: Dict[str, FAISS] = {}

        os.makedirs(faiss_base_path, exist_ok=True)

    def _get_faiss_path(self, session_id: str) -> str:
        safe_session_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return os.path.join(self.faiss_base_path, safe_session_id)

    def _get_or_create_short_term(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.short_term_memories:
            self.short_term_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                return_messages=True
            )
        return self.short_term_memories[session_id]

    def _get_or_create_long_term(self, session_id: str) -> FAISS:
        if session_id not in self.long_term_memories:
            faiss_path = self._get_faiss_path(session_id)
            if os.path.exists(faiss_path):
                self.long_term_memories[session_id] = FAISS.load_local(
                    folder_path=faiss_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE  
                )
            else:
                empty_doc = Document(page_content="initial empty document")
                self.long_term_memories[session_id] = FAISS.from_documents(
                    [empty_doc], self.embeddings, distance_strategy=DistanceStrategy.COSINE
                )
        return self.long_term_memories[session_id]

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted += f"{role}: {content}\n"
        return formatted.strip()

    def ingest(self, session_id: str, messages: List[Dict[str, str]]):
        short_term = self._get_or_create_short_term(session_id)

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                short_term.chat_memory.add_user_message(content)
            elif role == "assistant":
                short_term.chat_memory.add_ai_message(content)
            elif role == "system":
                short_term.chat_memory.add_user_message(f"[System] {content}")

        
        while len(short_term.chat_memory.messages) > self.max_short_term_messages:
            excess = len(short_term.chat_memory.messages) - self.max_short_term_messages
            old_messages = short_term.chat_memory.messages[:excess]
            old_history = [
                {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
                for msg in old_messages
            ]
            self._persist_old_to_long_term(session_id, old_history)
            del short_term.chat_memory.messages[:excess]

    def _persist_old_to_long_term(self, session_id: str, old_history: List[Dict[str, str]]):
        long_term = self._get_or_create_long_term(session_id)
        chat_text = self._format_messages(old_history)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([chat_text])

        long_term.add_documents(docs)
        faiss_path = self._get_faiss_path(session_id)
        long_term.save_local(faiss_path)


    def search(self, session_id: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
        long_term = self.long_term_memories.get(session_id, None)
        if long_term == None:
            return []
        results = long_term.similarity_search_with_score(query, k=k)
        passages = []
        for rs in results:
            
            rs = {
                "combined_information": rs[0].page_content,
                "score": rs[1]
            }
            passages.append(rs)
        
        scores, ranked_passages = reranker(query, passages)

        return [
            {"role": "long term history", "content": doc["combined_information"]}
            for doc, score in zip(ranked_passages, scores)
        ]

    def get_short_term_history(self, session_id: str) -> List[Dict[str, str]]:
        short_term = self._get_or_create_short_term(session_id)
        history = []
        for msg in short_term.chat_memory.messages:
            role = "user" if msg.type == "human" else "assistant"
            history.append({"role": role, "content": msg.content})
        return history

    def clear_session(self, session_id: str):
        if session_id in self.short_term_memories:
            del self.short_term_memories[session_id]

        faiss_path = self._get_faiss_path(session_id)
        if os.path.exists(faiss_path):
            import shutil
            shutil.rmtree(faiss_path)

        if session_id in self.long_term_memories:
            del self.long_term_memories[session_id]


# --------------------- Test ---------------------
if __name__ == "__main__":

    embedding_config = EmbeddingConfig(name="Alibaba-NLP/gte-multilingual-base")
    sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=embedding_config)

    reranker = Reranker(model_name="Alibaba-NLP/gte-multilingual-reranker-base")

    memory = MemoryModule(embedding_model=sentenceTransformerEmbedding, max_short_term_messages=2, reranker_model = reranker)

    
    memory.ingest("user_001", [
        {"role": "user", "content": "Hãy nhớ thông tin tôi tên là Dự"},
        {"role": "assistant", "content": "Chắc chắn rồi! tôi sẽ nhớ thông tin của bạn"}
    ])

    memory.ingest("user_001", [
        {"role": "user", "content": "Bạn có thể giúp tôi viết một bài luận không?"},
        {"role": "assistant", "content": "Chắc chắn rồi! Chủ đề là gì?"}
    ])

    memory.ingest("user_002", [
        {"role": "user", "content": "Tôi muốn viết bài luận về chủ đề làng của tôi"},
        {"role": "assistant", "content": "oke bây giờ tôi sẽ bắt đầu viết "}
    ])

    memory.ingest("user_002", [
        {"role": "user", "content": "Hôm nay trời khá là đẹp, bạn nghĩ tôi nên đi chơi không"},
        {"role": "assistant", "content": "oke tôi nghĩ ổn đấy"}
    ])

    results = memory.search("user_001", query="bài luận của tôi là gì", k=1)

    print("Short-term history:", memory.get_short_term_history("user_001"))
    print("Long-term history:", results)