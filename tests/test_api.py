import pytest
from PIL import Image
import io
from pathlib import Path
from starlette.testclient import TestClient
from app import app

def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200

def test_predict(client):
    with open("samples/Forest_9.jpg", "rb") as f:
        img_bytes = f.read()

    r = client.post(
        '/predict',
        files={'file': ("Forest_9.jpg", img_bytes, "image/jpeg")},
    )

    assert r.status_code == 200
    body = r.json()

    assert body['confidence'] == pytest.approx(0.9428, abs=1e-3)

    assert body['class_name'] == 'HerbaceousVegetation'

def test_predict_reject_non_image_content_type(client):
    content = b"This is a content"

    r = client.post(
        '/predict',
        files={'file': ("notes.txt", content, "text/plain")}
    )


    detail = r.json()["detail"]

    assert detail == "Declared content type doesn't start with image/"
    assert r.status_code == 415



def test_predict_reject_non_image_actual_content(client):
    content = b"This is a content"
    r = client.post(
        '/predict',
        files={'file':("notes.txt", content, "image/jpeg")}
    )

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail == "Uploaded file is not in the expected type"

def test_predict_reject_empty_file(client):
    content = b""
    r = client.post(
        '/predict',
        files={'file':("no-file", content, "image/jpeg")}
    )

    assert r.status_code == 400
    detail = r.json()['detail']
    assert detail == "Data is empty"



def test_predict_no_file(client):
    r = client.post('/predict')
    assert r.status_code == 422



def test_preprocess_grayscale_200(client):
    img = Image.new("L", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png_bytes = buf.getvalue()

    r = client.post(
        '/predict',
        files={'file' : ("gray.png", png_bytes, 'image/png')}
    )

    assert r.status_code == 200
    body = r.json()
    assert body['class_name'] == 'HerbaceousVegetation'



path = Path(__file__).resolve()
path = path.parent.parent
WEIGHTS_PATH = path / "models" / "best.pt"

FILE_PATH = path / "samples" / "Highway_1899.jpg"


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason='model weights not available')
def test_real_model_prediction():
    with TestClient(app) as c:
        with open(FILE_PATH, 'rb') as f:
            content = f.read()
        r = c.post(
            '/predict',
            files={'file' : ("Highway_1899.jpg", content, "image/jpeg")}
        )

    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] == "Highway"






