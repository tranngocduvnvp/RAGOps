
import requests

API_URL_reranker = "http://localhost:7777/rerank"

payload = {
  "requests": [
    {
      "data": [["Hoa màu gì?", "Nghiên cứu tài liệu thông quan LLM"], ["Hoa màu gì?", "tôi là model gemini 2.5 có khả năng đọc hiểu văn bản"], ["Hoa màu gì?", "hehe"], ["Hoa màu gì?", "hoa mau do"]]
    }
  ]
}

try:
    response = requests.post(API_URL_reranker, json=payload, timeout=10000)
    if response.status_code == 200:
        result = response.json()
        print("Scores:", result)
    else:
        print("Error:", response.text)
except Exception as e:
    print(e)