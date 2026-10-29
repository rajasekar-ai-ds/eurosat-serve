from fastapi import FastAPI, UploadFile, HTTPException, File
import asyncio
from contextlib import asynccontextmanager
from PIL import Image
from PIL import UnidentifiedImageError
from predict import load_model, preprocess, predict
import io
from pathlib import Path


@asynccontextmanager
async def lifespan(app:FastAPI):
    #startup
    path = Path(__file__).resolve()
    path = path.parent / "models"
    app.state.model, app.state.class_names = load_model(path / "best.pt", path / "classes.json")
    yield
    #shutdown
    #These are not mandatory as the app.state will be erased when the app shuts down.
    app.state.model = None
    app.state.class_names = None

app = FastAPI(lifespan=lifespan)

@app.get('/health')
def health():
    if getattr(app.state, "model", None) is not None:
        return {"status" : "okay"}
    else:
        raise HTTPException(status_code=503, detail="Model is not loaded")

@app.post('/predict')
async def predict_(file:UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(415, "Declared content type doesn't start with image/")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Data is empty")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Uploaded file is not in the expected type")

    model = app.state.model
    class_names = app.state.class_names

    x = preprocess(img)
    class_name, confidence = await asyncio.to_thread(predict, model, class_names, x)

    return {'class_name' : class_name , 'confidence' : round(confidence, 4), 'file_name' : file.filename}








