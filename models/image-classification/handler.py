import io
import json

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights
from ts.torch_handler.base_handler import BaseHandler


class ImageClassificationHandler(BaseHandler):

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.model = torch.jit.load(
            f"{model_dir}/model.pt",
            map_location=self.device,
        )

        self.model.eval()

        weights = ResNet18_Weights.DEFAULT
        self.transforms = weights.transforms()

        self.labels = weights.meta["categories"]

    def preprocess(self, data):
        images = []

        for row in data:
            image_bytes = row.get("data") or row.get("body")

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            image = self.transforms(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data):
        probabilities = torch.nn.functional.softmax(
            data,
            dim=1,
        )

        results = []

        for probs in probabilities:
            values, indices = torch.topk(probs, 5)

            predictions = []

            for value, index in zip(values, indices):
                predictions.append({
                    "class": self.labels[index.item()],
                    "confidence": round(value.item(), 4),
                })

            results.append(predictions)

        return results
