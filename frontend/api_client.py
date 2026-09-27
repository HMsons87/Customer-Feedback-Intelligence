import os
import sys
from pathlib import Path
import requests
from typing import Dict, Any, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

class FeedbackAPIClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
        self._local_service = None

    def _get_local_service(self):
        if self._local_service is None:
            try:
                from backend.services.model_service import model_service
                self._local_service = model_service
            except Exception as e:
                print(f"Error loading local model service: {e}")
        return self._local_service

    def check_health(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=2)
            if res.status_code == 200:
                return True
        except Exception:
            pass

        local_svc = self._get_local_service()
        if local_svc and (local_svc.pipeline is not None or local_svc.classifier is not None):
            return True

        return False

    def analyze_feedback(
        self,
        text: str,
        age: int = 35,
        department: str = "Dresses"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/analyze",
                json={
                    "review_text": text,
                    "age": age,
                    "department_name": department
                },
                timeout=5
            )
            if response.status_code == 200:
                return response.json(), None
        except Exception:
            pass

        local_svc = self._get_local_service()
        if local_svc:
            try:
                res = local_svc.analyze_feedback(
                    raw_text=text,
                    age=age,
                    department=department
                )
                return res, None
            except Exception as e:
                return None, f"Local Engine Error: {str(e)}"

        return None, "Inference Engine Unavailable. Ensure backend is running or models are loaded."