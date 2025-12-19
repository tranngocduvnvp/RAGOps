# class Reflection():
#     def __init__(self, llm):
#         self.llm = llm

#     def _concat_and_format_texts(self, data):
#         concatenatedTexts = []
#         for entry in data:
#             role = entry.get('role', '')
#             if entry.get('parts'):
#                 all_texts = ' '.join(part['text'] for part in entry['parts'] )
#             elif entry.get('content'):
#                 all_texts = entry['content'] 
#             concatenatedTexts.append(f"{role}: {all_texts} \n")
#         return ''.join(concatenatedTexts)


#     def __call__(self, chatHistory, lastItemsConsidereds=1000):
        
#         if len(chatHistory) >= lastItemsConsidereds:
#             chatHistory = chatHistory[len(chatHistory) - lastItemsConsidereds:]

#         historyString = self._concat_and_format_texts(chatHistory)

#         higherLevelSummariesPrompt = {
#             "role": "user",
#             "content": """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question in Vietnamese which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. {historyString}
#         """.format(historyString=historyString)
#         }

#         print(higherLevelSummariesPrompt)

#         completion = self.llm.generate_content([higherLevelSummariesPrompt])
    
#         return completion
    

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
            ) if history else "Không có"

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
            return query["content"]
        else:
            # return query["content"]
            prompt_rewrite = self.build_query_rewrite_prompt( query, stm, ltm)

        print("start llm to generate rewrite")
        completion = self.llm.generate_content([{"role":"user", "content": prompt_rewrite}])
        print("completion:", completion)
    
        return completion


