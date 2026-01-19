# Base Node complète (bullseye)
FROM node:20-bullseye

WORKDIR /app

# Installer Python et dépendances système (NE RIEN SUPPRIMER, AJOUTER SEULEMENT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libavcodec-extra \
    build-essential \
    libatlas3-base \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    cmake \
    pkg-config \
    git \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier package.json et installer Node deps
COPY package*.json ./
RUN npm install

# Copier requirements Python
COPY requirements.txt .

# Upgrade pip tooling (ajout)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Installer Torch via wheel précompilé pour éviter build (tu l'avais déjà)
RUN pip3 install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances Python du requirements (AJOUT, ne remplace pas)
RUN pip3 install --no-cache-dir -r requirements.txt

# Installer le reste des dépendances Python (tu l'avais déjà) + ajouts utiles
RUN pip3 install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.25.0 \
    numpy==1.26.0 \
    scipy==1.11.0 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    crepe==0.0.16 \
    numba \
    resampy \
    audioread \
    pydub \
    matplotlib \
    scikit-learn

# Copier tout le projet
COPY . .

# Exposer le port pour Render
EXPOSE 3000

# Lancer Node comme process principal
CMD ["node", "server.js"]
