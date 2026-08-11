import io
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from ts.torch_handler.base_handler import BaseHandler

from model import PersonDetector


class PersonDetectionHandler(BaseHandler):

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        self.model_dir = properties.get("model_dir")

        self.device = torch.device("cpu")

        # Build architecture without downloading weights.
        self.model = PersonDetector()

        # Load local checkpoint.
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(
            self.model_dir,
            serialized_file,
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        with open(
            os.path.join(
                self.model_dir,
                "index_to_name.json",
            )
        ) as f:
            self.labels = json.load(f)

        self.initialized = True

        print("Person detector initialized successfully.")

    def preprocess(self, data):
        images = []

        for item in data:
            image_bytes = item.get("body")

            if image_bytes is None:
                image_bytes = item.get("data")

            if image_bytes is None:
                raise ValueError("No image data found in request.")

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            image_tensor = self.transform(image)

            images.append(image_tensor)

        return images

    def inference(self, images, *args, **kwargs):
        with torch.no_grad():
            return self.model(images)

    def postprocess(self, outputs):
        results = []

        for output in outputs:
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            detections = []

            for box, label, score in zip(
                boxes,
                labels,
                scores,
            ):
                label_id = int(label.item())
                confidence = float(score.item())

                class_name = self.labels.get(
                    str(label_id),
                    str(label_id),
                )

                # Only return persons.
                if class_name != "person":
                    continue

                # Don't return extremely weak detections.
                if confidence < 0.50:
                    continue

                detections.append({
                    "class": "person",
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "box": [
                        round(float(x), 2)
                        for x in box.tolist()
                    ],
                })

            results.append({
                "detections": detections,
                "count": len(detections),
            })

        return results
