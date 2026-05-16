FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless==4.13.0.92

# Copier le code
COPY . .

# Télécharger les modèles
RUN python download_models.py

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
