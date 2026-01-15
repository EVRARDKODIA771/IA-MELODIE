# Utiliser Python 3.10 slim
FROM python:3.10-slim

# Répertoire de travail à la racine du container
WORKDIR /app

# Installer dépendances système pour audio / compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libatlas3-base \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port utilisé par Python
EXPOSE 8000

# Lancer le serveur Python
CMD ["python3", "app.py"]
