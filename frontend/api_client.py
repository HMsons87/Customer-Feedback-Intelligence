import os
import sys
from pathlib import Path
import requests
from typing import Dict, Any, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "FeedbackIQ-StreamlitClient"
}


class FeedbackAPIClient:
    def __init__(self, base_url: Optional[str] = None):
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            backend_url = None
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
                    backend_url = st.secrets["BACKEND_URL"]
            except BaseException:
                backend_url = None

            chosen_url = backend_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
            self.base_url = chosen_url.rstrip("/")

        self._local_service = None

    def check_health(self) -> bool:
        try:
            res = requests.get(
                f"{self.base_url}/health",
                headers=NGROK_HEADERS,
                timeout=5
            )
            if res.status_code == 200 and "application/json" in res.headers.get("content-type", ""):
                return True
        except Exception:
            pass
        return False

    def analyze_feedback(
            self,
            text: str,
            age: int = 35,
            department: str = "Dresses"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """إرسال المراجعة إلى FastAPI واستلام توقع الموديل والـ XAI"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/analyze",
                json={
                    "review_text": text,
                    "age": age,
                    "department_name": department
                },
                headers=NGROK_HEADERS,
                timeout=20
            )

            # إذا استجاب السيرفر بنجاح
            if response.status_code == 200:
                try:
                    return response.json(), None
                except Exception:
                    return None, f"Server returned non-JSON response: {response.text[:150]}"

            # في حال وجود خطأ في كود الـ Backend
            try:
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_detail = response.text[:200] or f"HTTP {response.status_code}"

            return None, f"Backend Error: {error_detail}"

        except requests.exceptions.Timeout:
            return None, "Request Timed Out: Ngrok tunnel took too long to respond."
        except requests.exceptions.RequestException as e:
            return None, f"Tunnel Connection Failed: {str(e)}"