import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, THRESHOLD_LIGHT, THRESHOLD_HIGH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ToxicityModel")

class ToxicityPredictor:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.toxic_index = 1
        self._load_model()

    def _load_model(self):
        """Charge le tokenizer et le modèle CamemBERT de classification de toxicité."""
        try:
            logger.info(f"Chargement du modèle CamemBERT : {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self._identify_toxic_index()
            labels_map = getattr(self.model.config, "id2label", {})
            logger.info(f"Modèle prêt sur {self.device} | Labels : {labels_map} | Index Toxique : {self.toxic_index}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {e}")
            self.tokenizer = None
            self.model = None

    def _identify_toxic_index(self):
        """Identifie l'index de la classe toxique (Index 1 pour neutral=0, toxic=1)."""
        if not self.model or not hasattr(self.model.config, "id2label"):
            self.toxic_index = 1
            return

        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            label_str = str(label).lower().strip()
            if any(neg in label_str for neg in ["neutral", "non", "normal", "clean", "label_0"]):
                continue
            if any(pos in label_str for pos in ["toxic", "hate", "insult", "label_1"]):
                self.toxic_index = int(idx)
                return

        self.toxic_index = 1 if len(id2label) > 1 else 0

    def split_into_sentences(self, text: str) -> list[str]:

        if not text or not str(text).strip():
            return []

        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', str(text).strip())
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 3]

        return sentences if sentences else [str(text).strip()]

    def predict_sentence(self, sentence: str) -> dict:

        if not sentence or not str(sentence).strip():
            return {
                "sentence": "",
                "prediction": "non toxique",
                "category": "non_toxique",
                "score": 0.0,
                "p_toxic": 0.0
            }

        p_toxic = 0.0

        if self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                probabilities = torch.softmax(outputs.logits, dim=-1)[0]
                p_toxic = float(probabilities[self.toxic_index].item())

            except Exception as e:
                logger.error(f"Erreur lors de l'inférence de la phrase : {e}")
                p_toxic = 0.01

        p_toxic = float(p_toxic)

        #Catégorisation selon les seuils configurés
        if p_toxic >= THRESHOLD_HIGH:
            category = "tres_toxique"
            prediction = "toxique"
        elif p_toxic >= THRESHOLD_LIGHT:
            category = "legerement_toxique"
            prediction = "toxique"
        else:
            category = "non_toxique"
            prediction = "non toxique"

        return {
            "sentence": sentence,
            "prediction": prediction,
            "category": category,
            "score": round(p_toxic, 4),
            "p_toxic": round(p_toxic, 4)
        }

    #Point d'entrée principal pour l'API REST et les analyses :
    # Segmente l'article en phrases.
    # Prédit chaque phrase individuellement.
    # Agrémente les métriques globales (score moyen, score max, ratio de phrases toxiques).

    def predict(self, text: str) -> dict:
        sentences = self.split_into_sentences(text)

        if not sentences:
            return {
                "prediction": "non toxique",
                "toxicity_category": "non_toxique",
                "score": 0.0,
                "mean_toxicity": 0.0,
                "max_toxicity": 0.0,
                "toxic_chunk_ratio": 0.0,
                "number_of_chunks": 1,
                "sentences_predictions": []
            }

        sentence_preds = [self.predict_sentence(s) for s in sentences]
        scores = [sp["p_toxic"] for sp in sentence_preds]

        mean_score = sum(scores) / len(scores)
        max_score = max(scores)
        toxic_sentences_count = sum(1 for s in scores if s >= THRESHOLD_LIGHT)
        toxic_ratio = round(toxic_sentences_count / len(sentences), 4)

        #Règles de décision pour l'article global :
        # Très toxique : Présence d'une phrase très toxique (max >= THRESHOLD_HIGH) ou forte densité toxique
        # Légèrement toxique : Au moins une phrase dépassant THRESHOLD_LIGHT
        # Non toxique : Aucune phrase toxique
        if max_score >= THRESHOLD_HIGH or (toxic_sentences_count >= 2 and mean_score >= 0.30):
            category = "tres_toxique"
            prediction = "toxique"
        elif max_score >= THRESHOLD_LIGHT or toxic_sentences_count >= 1:
            category = "legerement_toxique"
            prediction = "toxique"
        else:
            category = "non_toxique"
            prediction = "non toxique"

        return {
            "prediction": prediction,
            "toxicity_category": category,
            "score": round(mean_score, 4),
            "mean_toxicity": round(mean_score, 4),
            "max_toxicity": round(max_score, 4),
            "toxic_chunk_ratio": toxic_ratio,
            "number_of_chunks": len(sentences),
            "sentences_predictions": sentence_preds
        }

# Instance globale 
predictor = ToxicityPredictor()