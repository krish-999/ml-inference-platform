import torch.nn as nn

from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class PersonDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
        )

    def forward(self, images):
        return self.model(images)
