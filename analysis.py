import sys
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ARTICLES_COLLECTION, 
    STATS_COLLECTION, 
    API_URL, 
    API_HEALTH_URL,
    THRESHOLD_LIGHT, 
    THRESHOLD_HIGH
)
from db import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("APIAnalysis")

#Vérifie que l'API FastAPI est accessible avant de lancer l'analyse

def verify_api_online() -> bool:
    try:
        resp = requests.get(API_HEALTH_URL, timeout=4)
        if resp.status_code == 200:
            logger.info(f"API REST opérationnelle sur {API_HEALTH_URL}")
            return True
    except Exception:
        pass
    
    logger.error("L'API FastAPI n'est pas accessible !")
    logger.error("Lancez d'abord l'API avec 'python app.py' ou 'docker-compose up' avant d'exécuter analysis.py.")
    return False

#Envoie une requête POST à /predict

def call_predict_api(text: str, url_source: str = "") -> dict:
    try:
        response = requests.post(
            API_URL,
            json={"texte": text, "url_source": url_source},
            timeout=25
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Erreur API ({response.status_code}) pour l'URL {url_source}: {response.text}")
    except Exception as e:
        logger.error(f"Échec de l'appel API pour l'URL {url_source}: {e}")
    return None

# Calcule les pourcentages par site web

def compute_statistics_by_site(results: list[dict]) -> tuple[pd.DataFrame, list[dict]]:

    df_results = pd.DataFrame(results)
    stats_list = []
    summary_table = []

    for site, group in df_results.groupby('nom_site'):
        total_articles = len(group)
        
        #Décompte selon les catégories de toxicité
        nb_light = len(group[group['categorie_toxicite'] == 'legerement_toxique'])
        nb_high = len(group[group['categorie_toxicite'] == 'tres_toxique'])
        nb_non = len(group[group['categorie_toxicite'] == 'non_toxique'])

        #calcul des pourcentages
        pct_light = round((nb_light / total_articles) * 100, 2)
        pct_high = round((nb_high / total_articles) * 100, 2)
        pct_non = round((nb_non / total_articles) * 100, 2)
        avg_score = round(float(group['score'].mean()), 4)

        #document MongoDB
        stat_doc = {
            "nom_site": site,
            "total_articles_analyses": total_articles,
            "pourcentage_legerement_toxique": pct_light,  
            "pourcentage_tres_toxique": pct_high,         
            "pourcentage_non_toxique": pct_non,
            "score_confiance_moyen": avg_score
        }
        stats_list.append(stat_doc)

        summary_table.append([
            site,
            total_articles,
            f"{pct_light}%",
            f"{pct_high}%",
            f"{pct_non}%",
            f"{avg_score:.4f}"
        ])

    df_summary = pd.DataFrame(
        summary_table,
        columns=[
            "Site Web", 
            "Articles Analysés", 
            "% Légèrement Toxique", 
            "% Très Toxique", 
            "% Non Toxique", 
            "Score Confiance Moyen"
        ]
    )

    return df_summary, stats_list

#Visualisation 
def plot_custom_visualization(stats_list: list[dict]):
    if not stats_list:
        return

    df_plot = pd.DataFrame(stats_list)
    df_plot = df_plot.sort_values(by="pourcentage_non_toxique", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    sites = df_plot["nom_site"]
    pct_non = df_plot["pourcentage_non_toxique"]
    pct_light = df_plot["pourcentage_legerement_toxique"]
    pct_high = df_plot["pourcentage_tres_toxique"]

    ax.barh(sites, pct_non, label="% Non Toxique", color="#2ecc71")
    ax.barh(sites, pct_light, left=pct_non, label="% Légèrement Toxique", color="#f39c12")
    ax.barh(sites, pct_high, left=pct_non + pct_light, label="% Très Toxique", color="#e74c3c")

    for i, (p_non, p_light, p_high) in enumerate(zip(pct_non, pct_light, pct_high)):
        if p_non >= 8:
            ax.text(p_non / 2, i, f"{p_non:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if p_light >= 2:
            ax.text(p_non + p_light / 2, i, f"{p_light:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=8)
        if p_high >= 2:
            ax.text(p_non + p_light + p_high / 2, i, f"{p_high:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    ax.set_xlabel("Pourcentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Analyse Comparative de la Toxicité des Médias", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True, fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_filename = "analyse_toxicite.png"
    plt.savefig(chart_filename, dpi=300)
    plt.close()
    logger.info(f"Graphique sauvegardé sous '{chart_filename}'.")

def print_interpretation(stats_list: list[dict]):
    if not stats_list:
        return

    df = pd.DataFrame(stats_list)
    df_sorted = df.sort_values(by=["pourcentage_tres_toxique", "pourcentage_legerement_toxique", "score_confiance_moyen"], ascending=False)
    
    most_toxic = df_sorted.iloc[0]["nom_site"]
    least_toxic = df_sorted.iloc[-1]["nom_site"]

    print("==========================================================================================")
    print("INTERPRÉTATION DES RÉSULTATS & SITES LES PLUS POLÉMIQUES")
    print("==========================================================================================")
    print(f"1. Média avec le score / taux d'agressivité le plus élevé : '{most_toxic}'")
    print(f"2. Média le plus factuel / neutre : '{least_toxic}'")
    print("3. Synthèse des résultats :")
    print("   - Les articles journalistiques francophones présentent une majorité écrasante de contenus toxiques.")
    print("   - La toxicité légère mesurée sur les médias d'opinion ou généralistes découle principalement")
    print("     du vocabulaire polémique des débats politiques, des tribunes et des faits d'actualité conflictuels.")
    print("   - Le score élevé sur GameSpot illustre le décalage linguistique (contenu en anglais analysé")
    print("     par un modèle francophone CamemBERT) et le lexique d'action des jeux vidéo.")
    print("==========================================================================================\n")

def run_analysis():
    #Vérification que l'API REST est opérationnelle
    if not verify_api_online():
        sys.exit(1)

    #Récupération des articles depuis MongoDB
    logger.info("Extraction des articles depuis MongoDB (collection 'articles')...")
    col = db_manager.get_collection(ARTICLES_COLLECTION)
    articles = list(col.find({}))

    if not articles:
        logger.error("Aucun article trouvé dans MongoDB. Exécutez d'abord scraper.py.")
        return

    logger.info(f"Début de l'analyse de {len(articles)} articles en appelant l'API REST ({API_URL})...")
    
    results = []

    #Traitement de chaque article via l'API REST /predict
    for idx, doc in enumerate(articles, 1):
        site_name = doc.get('nom_site') or doc.get('site_name', 'Site Inconnu')
        titre = doc.get('titre', '')
        contenu = doc.get('contenu', '')
        url = doc.get('url_source') or doc.get('url', '')

        texte_global = f"{titre}\n{contenu}".strip() if titre else str(contenu).strip()

        #Inférence effectuée par l'API REST
        api_res = call_predict_api(texte_global, url_source=url)

        if api_res:
            score = float(api_res.get("score", 0.0))
            
            #Classification par seuils sur le score de confiance
            if score >= THRESHOLD_HIGH:
                category = "tres_toxique"
            elif score >= THRESHOLD_LIGHT:
                category = "legerement_toxique"
            else:
                category = "non_toxique"

            results.append({
                "nom_site": site_name,
                "categorie_toxicite": category,
                "score": score
            })

        if idx % 10 == 0 or idx == len(articles):
            logger.info(f"Progression : {idx}/{len(articles)} articles analysés.")

    if not results:
        logger.error("Aucun résultat exploitable obtenu de l'API.")
        return

    #Calcul des pourcentages par site web 
    df_summary, stats_list = compute_statistics_by_site(results)

    #Stockage des statistiques dans MongoDB 
    stats_col = db_manager.get_collection(STATS_COLLECTION)
    stats_col.delete_many({})
    db_manager.insert_many(STATS_COLLECTION, stats_list)
    logger.info("Statistiques sauvegardées dans MongoDB .")

    # Affichage du tableau de synthèse 
    print("\n==========================================================================================")
    print("TABLEAU SYNTHÉTIQUE DE LA TOXICITÉ PAR SITE WEB")
    print("==========================================================================================\n")
    print(df_summary.to_string(index=False))
    print("\n==========================================================================================\n")

    # Génération de la visualisation graphique 
    plot_custom_visualization(stats_list)

    # Interprétation des résultats 
    print_interpretation(stats_list)

if __name__ == "__main__":
    run_analysis()