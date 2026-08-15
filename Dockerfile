FROM python:3.10-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Configuration des variables d'environnement
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

# Utilisation de sh -c pour que la variable d'environnement $PORT soit résolue dynamiquement
CMD ["sh", "-c", "uvicorn app:app --host $HOST --port $PORT"]