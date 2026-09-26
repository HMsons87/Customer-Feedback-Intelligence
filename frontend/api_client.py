import os
import requests
import streamlit as st
from typing import Dict, Any, Optional, Tuple


class FeedbackAPIClient:
    def __init__(self, base_url: Optional[str] = None):
        if base_url:
            self.base_url = base_url
        elif hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            self.base_url = st.secrets["BACKEND_URL"]
        else:
            self.base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    def check_health(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=5)
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
                timeout=15
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