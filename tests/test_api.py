from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Quest Tracker API 준비 완료!" in response.json()["message"]

def test_user_completed():
    response = client.get("/user/completed")
    assert response.status_code == 200
    assert "user_completed" in response.json()

def test_quest_completion_rate():
    response = client.get("/quest/completion_rate")
    assert response.status_code == 200
    assert "quest_completion_rate" in response.json()
