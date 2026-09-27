import os
import re
import string
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import nltk

for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

PIPELINE_PATH = MODELS_DIR / "enhanced_pipeline.pkl"
CLUSTERING_PATH = MODELS_DIR / "kmeans_complaints_model.pkl"


class ModelService:
    def __init__(self):
        self.pipeline = None
        self.clustering_model = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.is_ready = False

        self.cluster_names = {
            0: "Fit & Sizing Issues",
            1: "Fabric & Material Quality",
            2: "Design & Construction Flaws",
            3: "Logistics & Customer Support"
        }

        self.load_models()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words and len(w) > 2]
        return " ".join(tokens)

    def load_models(self):
        try:
            if PIPELINE_PATH.exists():
                self.pipeline = joblib.load(PIPELINE_PATH)
            else:
                print(f"[ModelService] Warning: Pipeline file not found at {PIPELINE_PATH}")

            if CLUSTERING_PATH.exists():
                self.clustering_model = joblib.load(CLUSTERING_PATH)
            else:
                print(f"[ModelService] Warning: Clustering model not found at {CLUSTERING_PATH}")

            if self.pipeline is not None:
                self.is_ready = True
                print("[ModelService] Models successfully loaded and ready.")
        except Exception as e:
            print(f"[ModelService] Error during model loading: {e}")
            self.is_ready = False

    def _extract_xai_drivers(self, cleaned_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        drivers = []
        if not cleaned_text or self.pipeline is None:
            return drivers

        try:
            preprocessor = self.pipeline.named_steps.get('preprocessor')
            classifier = self.pipeline.named_steps.get('classifier')

            if not preprocessor or not classifier:
                return drivers

            tfidf_step = preprocessor.named_transformers_.get('text')
            if not tfidf_step:
                return drivers

            feature_names = tfidf_step.get_feature_names_out()
            tfidf_vector = tfidf_step.transform([cleaned_text]).toarray()[0]
            non_zero_indices = np.where(tfidf_vector > 0)[0]

            if len(non_zero_indices) == 0:
                return drivers

            coefs = classifier.coef_[0]

            word_scores = []
            for idx in non_zero_indices:
                if idx < len(feature_names):
                    word = feature_names[idx]
                    score = tfidf_vector[idx] * coefs[idx]
                    word_scores.append((word, score))

            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            for word, score in word_scores[:top_n]:
                drivers.append({
                    "word": word,
                    "score": round(float(score), 3),
                    "direction": "Positive" if score > 0 else "Negative"
                })
        except Exception as e:
            print(f"[ModelService] XAI calculation note: {e}")

        return drivers

    def analyze_feedback(self, raw_text: str, age: int = 35, department: str = "Dresses") -> Dict[str, Any]:
        if not self.is_ready or self.pipeline is None:
            raise RuntimeError("Model pipeline is not initialized.")

        cleaned = self.clean_text(raw_text)
        review_len = len(raw_text)
        word_cnt = len(raw_text.split())

        input_df = pd.DataFrame([{
            "Review Text": cleaned,
            "Age": age,
            "Department Name": department,
            "Review Length": review_len,
            "Word Count": word_cnt
        }])

        probabilities = self.pipeline.predict_proba(input_df)[0]
        class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        pred_class_idx = int(np.argmax(probabilities))
        sentiment = class_mapping.get(pred_class_idx, "Neutral")
        confidence = float(probabilities[pred_class_idx])

        is_complaint = (sentiment == "Negative")
        complaint_cluster = None

        if is_complaint and self.clustering_model is not None:
            try:
                cluster_id = int(self.clustering_model.predict([cleaned])[0])
                complaint_cluster = {
                    "cluster_id": cluster_id,
                    "category_name": self.cluster_names.get(cluster_id, "General Dissatisfaction"),
                    "top_keywords": ["fit", "fabric", "cut", "quality", "size"]
                }
            except Exception:
                complaint_cluster = {
                    "cluster_id": 0,
                    "category_name": "Quality & Construction Defect",
                    "top_keywords": ["quality", "material", "durability"]
                }

        key_drivers = self._extract_xai_drivers(cleaned)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(float(probabilities[0]), 4),
                "Neutral": round(float(probabilities[1]), 4),
                "Positive": round(float(probabilities[2]), 4)
            },
            "is_complaint": is_complaint,
            "complaint_cluster": complaint_cluster,
            "key_drivers": key_drivers
        }


model_service = ModelService()