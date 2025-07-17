import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_get_prediction_valid(client):
    response = client.get("/predict/?years_experience=3.0&version=v1")
    assert response.status_code == 200
    assert "predicted_salary" in response.json