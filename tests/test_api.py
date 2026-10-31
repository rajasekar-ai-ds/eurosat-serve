import pytest

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
    with open("notes.txt") as f:
        content = f.read()

    r = client.post(
        '/predict',
        files={'file': ("notes.txt", content, "text/plain")}
    )


    detail = r.json()["detail"]

    assert detail == "Declared content type doesn't start with image/"
    assert r.status_code == 415



def test_predict_reject_non_image_actual_content(client):
    with open("notes.txt") as f:
        content = f.read()

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











