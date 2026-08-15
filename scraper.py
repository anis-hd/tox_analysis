import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin

# Import de la configuration avec MAX_ARTICLES_PER_SITE
from config import TARGET_SITES, ARTICLES_COLLECTION, MAX_ARTICLES_PER_SITE
from db import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProductionScraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
    "Referer": "https://www.google.com/"
}

def clean_text(raw_text: str) -> str:
    """Nettoie le texte en supprimant les doublons et les espaces superflus."""
    if not raw_text:
        return ""
    soup = BeautifulSoup(raw_text, 'html.parser')
    text = soup.get_text()
    
    text = text.replace('\xa0', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 30]
    
    seen = set()
    cleaned = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            cleaned.append(line)
            
    return " ".join(cleaned).strip()

def extract_article_content(url: str, fallback_summary: str = "") -> tuple[str, str]:
    """Scrape le titre et le contenu textuel d'un article."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            titre = ""
            h1 = soup.find('h1')
            if h1:
                titre = h1.get_text(strip=True)

            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
                tag.decompose()

            paragraphs = [p.get_text() for p in soup.find_all('p')]
            contenu = clean_text("\n".join(paragraphs))

            if contenu and len(contenu) > 120:
                return titre, contenu
    except Exception as e:
        logger.debug(f"Erreur d'extraction HTML pour {url}: {e}")

    clean_summary = clean_text(fallback_summary)
    return "", clean_summary

def fetch_articles_fallback_html(site_info: dict, max_articles: int = MAX_ARTICLES_PER_SITE) -> list[dict]:
    """Fallback de scraping HTML direct si le flux RSS est inaccessible."""
    site_name = site_info["name"]
    site_url = site_info["url"]
    domain = site_info["domain"]
    articles = []

    try:
        resp = requests.get(site_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = set()
            for a in soup.find_all('a', href=True):
                full_url = urljoin(site_url, a['href'])
                if domain in full_url and len(full_url) > len(site_url) + 5 and '-' in full_url:
                    if not any(x in full_url for x in ['/tag/', '/category/', '/auteur/', '/login', '/subscribe', 'facebook', 'twitter']):
                        links.add(full_url)
                if len(links) >= max_articles:
                    break

            for url in links:
                titre, contenu = extract_article_content(url)
                if contenu and len(contenu) > 120:
                    articles.append({
                        "nom_site": site_name,               
                        "url": url,                           
                        "titre": titre or "Titre non spécifié", 
                        "contenu": contenu                     
                    })
                time.sleep(0.2)
    except Exception as e:
        logger.error(f"Erreur fallback HTML sur {site_name}: {e}")

    return articles

def scrape_site_articles(site_info: dict, max_articles: int = MAX_ARTICLES_PER_SITE) -> list[dict]:
    """Scrape les articles récents d'un site web selon la limite configurée."""
    site_name = site_info["name"]
    domain = site_info["domain"]
    rss_url = site_info.get("rss_url", "")
    articles = []

    logger.info(f"[Collecte] Scraping de : {site_name} (Max: {max_articles} articles)...")

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            logger.warning(f" Flux RSS non joignable pour {site_name} (HTTP {resp.status_code}) -> Passage au scraping HTML direct")
            return fetch_articles_fallback_html(site_info, max_articles)

        soup = BeautifulSoup(resp.text, 'xml')
        items = soup.find_all('item')

        if not items:
            return fetch_articles_fallback_html(site_info, max_articles)

        for item in items[:max_articles]:
            title_tag = item.find('title')
            link_tag = item.find('link')
            desc_tag = item.find('description')

            titre = clean_text(title_tag.text) if title_tag else "Titre non spécifié"
            url = link_tag.text.strip() if link_tag else ""
            summary = desc_tag.text if desc_tag else ""

            if url:
                extracted_title, contenu = extract_article_content(url, summary)
                final_title = extracted_title if extracted_title else titre
                
                if contenu and len(contenu) > 80:
                    articles.append({
                        "nom_site": site_name,               
                        "url": url,                           
                        "titre": final_title,                
                        "contenu": contenu                  
                    })
                    logger.info(f"[{site_name}] Article {len(articles)} récupéré: {final_title[:45]}...")
            time.sleep(0.2)

    except Exception as e:
        logger.error(f"Erreur lors du scraping de {site_name}: {e}")
        return fetch_articles_fallback_html(site_info, max_articles)

    return articles

def run_pipeline():
    """Exécute la collecte et stocke DIRECTEMENT les articles dans MongoDB."""
    print("\n======================================================================")
    print(f"DÉMARRAGE DE LA COLLECTE (Limite configurée : {MAX_ARTICLES_PER_SITE} articles/site)")
    print("======================================================================\n")

    articles_col = db_manager.get_collection(ARTICLES_COLLECTION)
    
    # Réinitialisation de la collection pour repartir sur une base propre avec les clés fr
    articles_col.delete_many({})
    
    total_inserted = 0

    for site in TARGET_SITES:
        articles = scrape_site_articles(site, max_articles=MAX_ARTICLES_PER_SITE)
        
        if articles:
            inserted_ids = db_manager.insert_many(ARTICLES_COLLECTION, articles)
            count = len(inserted_ids)
            total_inserted += count
            logger.info(f" {count} articles de {site['name']} directement enregistrés dans MongoDB.\n")

    print("======================================================================")
    print(f"PIPELINE DE COLLECTE TERMINÉ AVEC SUCCÈS !")
    print(f"Total d'articles directement stockés dans MongoDB : {total_inserted}")
    print(f"Base de données : toxicity_db | Collection : {ARTICLES_COLLECTION}")
    print("======================================================================\n")

if __name__ == "__main__":
    run_pipeline()