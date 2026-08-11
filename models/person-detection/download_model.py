import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

model = ssdlite320_mobilenet_v3_large(weights=weights)

torch.save(
    model.state_dict(),
    "models/person-detection/ssdlite320_mobilenet_v3_large.pth",
)

print("Model weights saved.")
