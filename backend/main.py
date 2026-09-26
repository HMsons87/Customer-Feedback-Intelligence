from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import ReviewRequest, FeedbackAnalysisResponse
from services.model_service import model_service

app = FastAPI(
    title="Customer Feedback Intelligence Engine",
    description="Production API for NLP Sentiment Analysis & Unsupervised Complaints Clustering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    pipeline_ready = getattr(model_service, "pipeline", None) is not None
    classifier_ready = getattr(model_service, "classifier", None) is not None
    vectorizer_ready = getattr(model_service, "vectorizer", None) is not None
    clustering_ready = getattr(model_service, "kmeans", None) is not None

    engine_ready = pipeline_ready or (classifier_ready and vectorizer_ready)

    return {
        "status": "healthy" if engine_ready else "degraded",
        "engine_ready": engine_ready,
        "pipeline_ready": pipeline_ready,
        "clustering_ready": clustering_ready
    }


@app.post("/api/v1/analyze", response_model=FeedbackAnalysisResponse)
def analyze_review(payload: ReviewRequest):
    if not payload.review_text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    try:
        results = model_service.analyze_feedback(
            raw_text=payload.review_text,
            age=payload.age,
            department=payload.department_name
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))