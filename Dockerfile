FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless==4.13.0.92

COPY . .

RUN python download_models.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
