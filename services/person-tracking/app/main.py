import io
import math

import requests
from fastapi import FastAPI, File, UploadFile
from PIL import Image


app = FastAPI(title="Person Tracking Service")


TORCHSERVE_URL = (
    "http://torchserve:8080/predictions/person_detector"
)


class CentroidTracker:
    def __init__(self, max_distance=100):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance

    def centroid(self, box):
        x1, y1, x2, y2 = box

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    def distance(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
        )

    def update(self, detections):
        current_centroids = [
            self.centroid(d["box"])
            for d in detections
        ]

        updated_tracks = {}
        used_ids = set()

        for detection, centroid in zip(
            detections,
            current_centroids,
        ):
            best_id = None
            best_distance = self.max_distance

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue

                distance = self.distance(
                    centroid,
                    track["centroid"],
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            updated_tracks[best_id] = {
                "centroid": centroid,
                "box": detection["box"],
                "confidence": detection["confidence"],
            }

            used_ids.add(best_id)

        self.tracks = updated_tracks

        results = []

        for track_id, track in self.tracks.items():
            results.append({
                "track_id": track_id,
                "class": "person",
                "confidence": track["confidence"],
                "box": track["box"],
            })

        return results


tracker = CentroidTracker()


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/track")
async def track(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    # Validate that the input is an image.
    Image.open(
        io.BytesIO(image_bytes)
    ).verify()

    response = requests.post(
        TORCHSERVE_URL,
        data=image_bytes,
        headers={
            "Content-Type": "application/octet-stream"
        },
        timeout=60,
    )

    response.raise_for_status()

    detection_result = response.json()

    if isinstance(detection_result, dict):
        detections = detection_result["detections"]
    else:
        detections = detection_result[0]["detections"]

    tracks = tracker.update(
        detections
    )

    return {
        "tracks": tracks,
        "count": len(tracks),
    }
