from fastapi.testclient import TestClient
from fastapi import FastAPI
from api import app
import pytest 

client = TestClient(app)

@pytest.fixture
def test_image():
    with open("123.jpg", "rb") as f:
        image_bytes = f.read()
    return image_bytes

@pytest.mark.asyncio
def test_answer(test_image):
    response = client.post("/answer", files={"file" : ("test_image.png", test_image, "image/png")})
    assert response.status_code == 200
    assert "answer" in response.text

@pytest.mark.asyncio
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
