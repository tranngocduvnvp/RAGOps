
class Reflection():
    def __init__(self, llm):
        self.llm = llm

    def build_query_rewrite_prompt(self, query: dict, stm: list, ltm: list) -> str:
        """
        query: {"role": "user", "content": "..."}
        stm:   [{"role": "...", "content": "..."}, ...]
        ltm:   [{"role": "long term history", "content": "..."}]
        """

        def format_history(history):
            return "\n".join(
                f"{h['role'].capitalize()}: {h['content']}" for h in history
            ) if history else "Không có lịch sử chat"

        stm_text = format_history(stm)
        ltm_text = format_history(ltm)

        prompt = f"""
    Bạn là một AI chuyên thực hiện QUERY REWRITING cho hệ thống RAG.

    Nhiệm vụ:
    - Viết lại câu hỏi hiện tại của người dùng thành một truy vấn độc lập, rõ ràng, đầy đủ ngữ cảnh.
    - Truy vấn sau khi viết lại phải tối ưu cho semantic search / vector search.
    - Không trả lời câu hỏi.
    - Không giải thích.
    - Chỉ trả về MỘT câu truy vấn đã được viết lại.

    Short-term history:
    {stm_text}

    Long-term history:
    {ltm_text}

    Câu hỏi hiện tại của người dùng:
    User: {query['content']}

    Truy vấn đã được viết lại:
    """.strip()

        return prompt


    def __call__(self, query, stm, ltm):
        
        
        if len(stm)==0:
            return query["content"], {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        else:
            prompt_rewrite = self.build_query_rewrite_prompt( query, stm, ltm)

        completion, usage_dict = self.llm.generate_content([{"role":"user", "content": prompt_rewrite}])

        print(completion)
        print(usage_dict)
        return completion, usage_dict


