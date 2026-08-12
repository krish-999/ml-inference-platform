import requests

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Image Classification Service")

TORCHSERVE_URL = (
    "http://torchserve:8080/predictions/image_classifier"
)


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()

    response = requests.post(
        TORCHSERVE_URL,
        data=image_bytes,
        headers={
            "Content-Type": "application/octet-stream"
        },
        timeout=60,
    )

    response.raise_for_status()

    return {
        "predictions": response.json()
    }
