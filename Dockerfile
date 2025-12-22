# Sử dụng image Python 3.12.9 chính thức
FROM python:3.12.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cập nhật hệ thống và cài đặt các gói phụ thuộc cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install langfuse
# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Lệnh chạy mặc định (có thể thay đổi tùy ứng dụng)
CMD ["python", "serve.py"]
