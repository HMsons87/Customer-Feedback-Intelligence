import requests
from typing import Dict, Any, Optional, Tuple


class FeedbackAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url

    def check_health(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=3)
            return res.status_code == 200
        except requests.exceptions.RequestException:
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
                timeout=10
            )
            if response.status_code == 200:
                return response.json(), None

            try:
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"

            return None, f"Server Error: {error_detail}"

        except requests.exceptions.RequestException as e:
            return None, f"Connection Failed: {str(e)}"