import json
from chonkie import SemanticChunker




splitter = SemanticChunker(
    chunk_size=800,       # số ký tự trong 1 chunk
    overlap_size=100,     # overlap để không mất ngữ cảnh khi hỏi RAG
    use_sections=True     # tự tách theo heading (1., 2., 3., …)
)

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def get_price(html):
    import re

    #html = '<span class="woocommerce-Price-amount amount"><bdi>1.500.000<span class="woocommerce-Price-currencySymbol">₫</span></bdi></span>'

    match = re.search(r'<bdi>(.*?)<span', html)
    if match:
        price_number = match.group(1).strip()
        # print(price_number)
        return price_number

    
chunks = []

for item in data:
    text = item["content"]

    # tạo semantic chunks
    docs = splitter.chunk(text)

    for i, doc in enumerate(docs):
        chunks.append({
            "id": f"{item['title']}-{i}",
            "text": doc,
            "title": item["title"],
            # "metadata": {
                "url": item["url"],
                # "price": item["price"],
                "price": get_price(item["price"]),
                "image_urls": item["image_urls"],
            # }
        })



import csv



import csv

def save_chunks_to_csv(chunks, filename="chunks.csv"):
    # Extract all keys automatically
    keys = set()
    for c in chunks:
        keys.update(c.keys())
    keys = list(keys)

    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(chunks)

# Example usage:
save_chunks_to_csv(chunks, "chunks_datab.csv")
