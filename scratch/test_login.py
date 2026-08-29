import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("Starting test...")
# Trigger startup events
try:
    with client:
        # Try login
        response = client.post("/auth/login", json={
            "email": "admin@example.com",
            "password": "admin123"
        })
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.json())
except Exception as e:
    import traceback
    traceback.print_exc()
