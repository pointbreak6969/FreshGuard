from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import requests, io, os, uuid
from detection import detect
from classification import get_classified_result



app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    if not file and not url:
        return JSONResponse({"error": "Provide either an image file or a URL"}, status_code=400)

    if file:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    else:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    image_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    image.save(original_path)

    detections, annotated_path = detect(image, os.path.join(UPLOAD_DIR, image_id))

    if os.path.exists(original_path):
        os.remove(original_path)
    if os.path.exists(annotated_path):
        os.remove(annotated_path)

    return JSONResponse({"image_id": image_id, "original": original_path, "detections": detections})
