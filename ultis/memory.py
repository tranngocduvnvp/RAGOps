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
from embeddings import BaseEmbedding, EmbeddingConfig, SentenceTransformerEmbedding





class MemoryModule:
    """
    Memory module hỗ trợ multi-session cho RAG, sử dụng SentenceTransformer local cho embedding.
    - Short Term: ConversationBufferMemory riêng theo session.
    - Long Term: FAISS riêng theo session, lưu trên disk.
    """

    def __init__(
        self,
        embedding_model,
        reranker_model,
        faiss_base_path: str = "faiss_indexes",
        max_short_term_messages: int = 2,
    ):
        """
        :param embedding_config: Cấu hình model SentenceTransformer (ví dụ: EmbeddingConfig(name="vinai/phobert-base-v2"))
        :param faiss_base_path: Thư mục lưu các FAISS index theo session.
        :param max_short_term_messages: Số tin nhắn tối đa trong short-term trước khi persist.
        """
        # Khởi tạo embedding local
        self.embeddings = embedding_model
        self.reranker_model = reranker_model

        self.faiss_base_path = faiss_base_path
        self.max_short_term_messages = max_short_term_messages

        # Cache memory theo session_id
        self.short_term_memories: Dict[str, ConversationBufferMemory] = {}
        self.long_term_memories: Dict[str, FAISS] = {}

        # Tạo thư mục gốc
        os.makedirs(faiss_base_path, exist_ok=True)

    def _get_faiss_path(self, session_id: str) -> str:
        """Tạo đường dẫn an toàn cho FAISS index của session."""
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
                # Load FAISS đã lưu (với cosine similarity mặc định nếu normalize)
                self.long_term_memories[session_id] = FAISS.load_local(
                    folder_path=faiss_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE  # vì chúng ta normalize embeddings
                )
            else:
                # Tạo FAISS rỗng
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

    def ingress(self, session_id: str, messages: List[Dict[str, str]]):
        """
        Thêm tin nhắn vào short-term memory của session.
        Nếu vượt ngưỡng → tự động persist sang long-term.
        """
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
            # Tính số excess (thường là 1 hoặc vài, nhưng để an toàn dùng while)
            excess = len(short_term.chat_memory.messages) - self.max_short_term_messages
            # Lấy excess messages cũ nhất
            old_messages = short_term.chat_memory.messages[:excess]
            # Format thành dict cho _format_messages
            old_history = [
                {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
                for msg in old_messages
            ]
            self._persist_old_to_long_term(session_id, old_history)
            # Xóa excess old messages khỏi short-term
            del short_term.chat_memory.messages[:excess]

    def _persist_old_to_long_term(self, session_id: str, old_history: List[Dict[str, str]]):
        """Chuyển một batch old messages sang long-term FAISS."""
        long_term = self._get_or_create_long_term(session_id)

        # Format batch old thành text
        chat_text = self._format_messages(old_history)

        # Split thành chunks nếu cần
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([chat_text])

        # Thêm vào FAISS
        long_term.add_documents(docs)

        # Lưu xuống disk
        faiss_path = self._get_faiss_path(session_id)
        long_term.save_local(faiss_path)


    def search(self, session_id: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Tìm kiếm trong long-term memory của session."""
        # long_term = self._get_or_create_long_term(session_id)
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
        """Lấy lịch sử short-term dưới format OpenAI."""
        short_term = self._get_or_create_short_term(session_id)
        history = []
        for msg in short_term.chat_memory.messages:
            role = "user" if msg.type == "human" else "assistant"
            history.append({"role": role, "content": msg.content})
        return history

    def clear_session(self, session_id: str):
        """Xóa hoàn toàn memory của một session."""
        if session_id in self.short_term_memories:
            del self.short_term_memories[session_id]

        faiss_path = self._get_faiss_path(session_id)
        if os.path.exists(faiss_path):
            import shutil
            shutil.rmtree(faiss_path)

        if session_id in self.long_term_memories:
            del self.long_term_memories[session_id]


# --------------------- Ví dụ sử dụng ---------------------
if __name__ == "__main__":

    
    # Định nghĩa config cho model embedding (ví dụ: dùng model tiếng Việt)
    embedding_config = EmbeddingConfig(name="Alibaba-NLP/gte-multilingual-base")
    sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=embedding_config)

    reranker = Reranker(model_name="Alibaba-NLP/gte-multilingual-reranker-base")

    memory = MemoryModule(embedding_model=sentenceTransformerEmbedding, max_short_term_messages=2, reranker_model = reranker)

    
    memory.ingress("user_001", [
        {"role": "user", "content": "Hãy nhớ thông tin tôi tên là Dự"},
        {"role": "assistant", "content": "Chắc chắn rồi! tôi sẽ nhớ thông tin của bạn"}
    ])

    memory.ingress("user_001", [
        {"role": "user", "content": "Bạn có thể giúp tôi viết một bài luận không?"},
        {"role": "assistant", "content": "Chắc chắn rồi! Chủ đề là gì?"}
    ])

    memory.ingress("user_002", [
        {"role": "user", "content": "Tôi muốn viết bài luận về chủ đề làng của tôi"},
        {"role": "assistant", "content": "oke bây giờ tôi sẽ bắt đầu viết "}
    ])

    memory.ingress("user_002", [
        {"role": "user", "content": "Hôm nay trời khá là đẹp, bạn nghĩ tôi nên đi chơi không"},
        {"role": "assistant", "content": "oke tôi nghĩ ổn đấy"}
    ])

    # Tìm kiếm trong lịch sử
    results = memory.search("user_001", query="bài luận của tôi là gì", k=1)

    print("Short-term history:", memory.get_short_term_history("user_001"))
    print("Long-term history:", results)