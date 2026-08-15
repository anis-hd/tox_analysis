import logging
from datetime import datetime, timezone
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import PORT, HOST, PREDICTIONS_COLLECTION
from db import db_manager
from model import predictor

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ToxicityAPI")

app = FastAPI(
    title="API REST de Détection de Toxicité",
    description="API REST d'inférence NLP (CamemBERT) évaluant la toxicité textuelle et historisant les prédictions dans MongoDB.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

#validation des données I/O


class PredictRequest(BaseModel):
    texte: str = Field(
        ..., 
        min_length=2, 
        description="Contenu textuel à analyser pour détecter la toxicité.",
        example="Foutez le camp d'ici, vous êtes incompétents !"
    )
    url_source: Optional[str] = Field(
        default=None, 
        description="URL source de l'article ou du commentaire (optionnel).",
        example="https://www.lemonde.fr/"
    )

class PredictResponse(BaseModel):
    texte: str = Field(..., description="Texte analysé.")
    prediction: str = Field(..., description="Classification binaire : 'toxique' ou 'non toxique'.")
    score: float = Field(..., description="Score de confiance associé à la prédiction (entre 0.0 et 1.0).")


#Endpoints de l'API REST

@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Vérification de l'état du service",
    status_code=status.HTTP_200_OK
)
def health_check():
    """Retourne l'état de l'API et l'horodatage courant."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inférence"],
    summary="Prédire la toxicité d'un texte et enregistrer le résultat"
)
def predict_toxicity(request: PredictRequest):
    """
    Endpoint principal /predict :
    1. Valide les données entrantes via Pydantic.
    2. Découpe le texte en phrases et calcule la toxicité via CamemBERT.
    3. Enregistre la prédiction dans MongoDB avec des clés 100% en français (Partie 2.3).
    4. Retourne uniquement texte, prediction et score.
    """
    try:
        #Inférence NLP
        res = predictor.predict(request.texte)
        prediction_date = datetime.now(timezone.utc)

        #Construction du document MongoDB
        doc_prediction = {
            "texte": request.texte,
            "prediction": res["prediction"],
            "score": float(res["score"]),
            "date": prediction_date,
            "url_source": request.url_source,
            "toxicite_moyenne": float(res.get("mean_toxicity", res["score"])),
            "toxicite_max": float(res.get("max_toxicity", res["score"])),
            "ratio_phrases_toxiques": float(res.get("toxic_chunk_ratio", 0.0)),
            "nombre_de_phrases": int(res.get("number_of_chunks", 1))
        }

        #Sauvegarde dans MongoDB 
        inserted_id = db_manager.insert_one(PREDICTIONS_COLLECTION, doc_prediction)
        
        if inserted_id:
            logger.info(f"Inférence enregistrée dans MongoDB (ID: {inserted_id}) | '{res['prediction']}' (Score: {res['score']})")
        else:
            logger.warning("Attention! : l'enregistrement dans MongoDB a échoué.")

        #Réponse 
        return PredictResponse(
            texte=request.texte,
            prediction=res["prediction"],
            score=float(res["score"])
        )

    except Exception as e:
        logger.error(f"Erreur interne lors du traitement de la requête : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de l'inférence : {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)