# 使用官方 Python slim 镜像（支持 linux/amd64 与 linux/arm64）
# 验收可在苹果电脑上使用: docker pull --platform linux/arm64 python:3.11-slim
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py .

RUN mkdir -p /app/output

ENTRYPOINT ["python", "main.py"]
CMD ["--output-dir", "/app/output"]
