# Base Node + Python
FROM node:20-slim

# Répertoire de travail
WORKDIR /app

# Installer Python et dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libatlas3-base \
    && rm -rf /var/lib/apt/lists/*

# Copier package.json et package-lock.json pour Node
COPY package*.json ./

# Installer dépendances Node
RUN npm install

# Copier requirements Python et installer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copier tout le projet (Node + Python + modules)
COPY . .

# Exposer le port pour Render (Node écoute sur 3000)
EXPOSE 3000

# Lancer Node comme processus principal
CMD ["node", "server.js"]
