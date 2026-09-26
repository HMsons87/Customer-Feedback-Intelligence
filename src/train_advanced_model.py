import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_reviews.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Feature Engineering
df["Review_Length"] = df["Review Text"].fillna("").astype(str).str.len()
df["Word_Count"] = df["Review Text"].fillna("").astype(str).str.split().str.len()

def encode_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    return 2      # Positive

df["Target"] = df["Rating"].apply(encode_sentiment)

# Selected Features
text_feature = "Review Text"
num_features = ["Age", "Review_Length", "Word_Count"]
cat_features = ["Department Name"]

X = df[[text_feature] + num_features + cat_features]
y = df["Target"].values

# 2. Build Preprocessor (Feature Fusion)
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"), text_feature),
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# 3. Build Full Pipeline
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42))
])

# 4. Train-Test Split & Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training Enhanced Feature-Fusion Pipeline...")
full_pipeline.fit(X_train, y_train)

# 5. Save Model Pipeline
model_save_path = MODELS_DIR / "enhanced_pipeline.pkl"
joblib.dump(full_pipeline, model_save_path)
print(f"Pipeline successfully saved to {model_save_path}")