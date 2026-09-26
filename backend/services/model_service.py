import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.pipeline_path = models_dir / "enhanced_pipeline.pkl"
        self.cluster_path = models_dir / "kmeans_complaints_model.pkl"

        self.pipeline = self._load_file(str(self.pipeline_path))
        self.kmeans = self._load_file(str(self.cluster_path))

        self.cluster_labels = {
            0: {"name": "Fit & Sizing Issues", "keywords": ["small", "tight", "size", "large", "fit", "bust"]},
            1: {"name": "Fabric & Material Quality",
                "keywords": ["fabric", "cheap", "thin", "material", "wash", "color"]},
            2: {"name": "Design & Construction Flaws",
                "keywords": ["zipper", "seam", "armholes", "neck", "cut", "buttons"]},
            3: {"name": "Return & Order Fulfillment",
                "keywords": ["return", "store", "order", "package", "back", "customer"]}
        }

    def _load_file(self, path: str):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def explain_text_drivers(self, text: str, predicted_class: int, top_n: int = 5):
        """Explainable AI: حساب أوزان الكلمات المساهمة في قرار الموديل"""
        if not self.pipeline:
            return []

        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            classifier = self.pipeline.named_steps["classifier"]
            tfidf = preprocessor.named_transformers_["text"]

            # استخراج الكلمات المفتاحية وأوزانها
            feature_names = tfidf.get_feature_names_out()
            transformed_text = tfidf.transform([text.lower()])

            non_zero_indices = transformed_text.nonzero()[1]
            weights = classifier.coef_[predicted_class]

            drivers = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                score = transformed_text[0, idx] * weights[idx]

                direction = "Positive" if score > 0 else "Negative"
                if abs(score) > 0.05:
                    drivers.append({
                        "word": word,
                        "score": round(float(score), 4),
                        "direction": direction
                    })

            # ترتيب الكلمات بالأعلى تأثيراً
            drivers = sorted(drivers, key=lambda x: abs(x["score"]), reverse=True)[:top_n]
            return drivers
        except Exception:
            return []

    def analyze_feedback(self, raw_text: str, age: int = 35, department: str = "Dresses"):
        if not self.pipeline:
            raise RuntimeError("Pipeline model is not loaded. Train and save enhanced_pipeline.pkl first.")

        # تجهيز الداتا فريم المدخل بجميع الخصائص
        input_df = pd.DataFrame([{
            "Review Text": raw_text,
            "Age": age,
            "Review_Length": len(raw_text),
            "Word_Count": len(raw_text.split()),
            "Department Name": department
        }])

        pred = int(self.pipeline.predict(input_df)[0])
        probs = self.pipeline.predict_proba(input_df)[0]

        labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = labels_map.get(pred, "Neutral")
        confidence = float(max(probs))

        prob_dict = {
            "Negative": round(float(probs[0]), 4),
            "Neutral": round(float(probs[1]), 4) if len(probs) > 2 else 0.0,
            "Positive": round(float(probs[-1]), 4)
        }

        # تشغيل XAI لاستخراج الكلمات المؤثرة
        key_drivers = self.explain_text_drivers(raw_text, predicted_class=pred)

        # استخراج فئة الشكوى عبر K-Means إن كانت سلبية
        cluster_info = None
        is_complaint = (sentiment == "Negative")
        if is_complaint and self.kmeans:
            try:
                tfidf = self.pipeline.named_steps["preprocessor"].named_transformers_["text"]
                text_vec = tfidf.transform([raw_text])
                cluster_id = int(self.kmeans.predict(text_vec)[0])
                meta = self.cluster_labels.get(cluster_id, {"name": "General Defect", "keywords": []})
                cluster_info = {
                    "cluster_id": cluster_id,
                    "category_name": meta["name"],
                    "top_keywords": meta["keywords"]
                }
            except Exception:
                pass

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "is_complaint": is_complaint,
            "complaint_cluster": cluster_info,
            "key_drivers": key_drivers
        }


model_service = ModelService()