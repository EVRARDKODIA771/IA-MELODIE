# Base Node complète (bullseye)
FROM node:20-bullseye

WORKDIR /app

# Installer Python et dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libatlas3-base \
    libffi-dev \
    libssl-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier package.json et installer Node deps
COPY package*.json ./
RUN npm install

# Copier requirements Python
COPY requirements.txt .

# Installer Torch via wheel précompilé pour éviter build
RUN pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Installer le reste des dépendances Python
RUN pip3 install --no-cache-dir fastapi==0.109.0 uvicorn[standard]==0.25.0 numpy==1.26.0 scipy==1.11.0 librosa==0.10.1 soundfile==0.12.1 crepe==0.0.16

# Copier tout le projet
COPY . .

# Exposer le port pour Render
EXPOSE 3000

# Lancer Node comme process principal
CMD ["node", "server.js"]
