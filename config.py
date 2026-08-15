import os
from dotenv import load_dotenv

load_dotenv()

# Configuration MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "toxicity_db")

ARTICLES_COLLECTION = "articles"
PREDICTIONS_COLLECTION = "predictions"
STATS_COLLECTION = "toxicity_stats"

# Configuration Réseau / API
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Point d'accès REST pour l'analyse
API_HOST = "localhost" if HOST in ["0.0.0.0", "127.0.0.1"] else HOST
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{PORT}/predict")
API_HEALTH_URL = os.getenv("API_HEALTH_URL", f"http://{API_HOST}:{PORT}/health")

# Modèle NLP & Seuils
MODEL_NAME = "EIStakovskii/french_toxicity_classifier_plus_v2"
THRESHOLD_LIGHT = float(os.getenv("THRESHOLD_LIGHT", 0.70))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", 0.90))


# Nombre d'articles à récupérer par site (Modifiable ici)
MAX_ARTICLES_PER_SITE = int(os.getenv("MAX_ARTICLES_PER_SITE", 15))

# Liste des 8 médias cibles
TARGET_SITES = [
    {"name": "L'Humanité", "url": "https://www.humanite.fr/", "rss_url": "https://www.humanite.fr/feed", "domain": "humanite.fr"},
    {"name": "GameSpot", "url": "https://www.gamespot.com/", "rss_url": "https://www.gamespot.com/feeds/mashup/", "domain": "gamespot.com"},
    {"name": "Marianne", "url": "https://www.marianne.net/", "rss_url": "https://www.marianne.net/rss.xml", "domain": "marianne.net"},
    {"name": "Le Monde", "url": "https://www.lemonde.fr/", "rss_url": "https://www.lemonde.fr/rss/une.xml", "domain": "lemonde.fr"},
    {"name": "France 24", "url": "https://www.france24.com/fr/", "rss_url": "https://www.france24.com/fr/rss", "domain": "france24.com"},
    {"name": "France 3 Régions", "url": "https://france3-regions.franceinfo.fr/", "rss_url": "https://france3-regions.franceinfo.fr/actu/rss", "domain": "france3-regions.franceinfo.fr"},
    {"name": "Médiacités", "url": "https://www.mediacites.fr/", "rss_url": "https://www.mediacites.fr/feed/", "domain": "mediacites.fr"},
    {"name": "Le Point", "url": "https://www.lepoint.fr/24h-infos/", "rss_url": "https://www.lepoint.fr/24h-infos/rss.xml", "domain": "lepoint.fr"},
]