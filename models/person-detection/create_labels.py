import json
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

categories = weights.meta["categories"]

mapping = {
    str(index): name
    for index, name in enumerate(categories)
}

with open(
    "models/person-detection/index_to_name.json",
    "w",
) as f:
    json.dump(mapping, f, indent=2)

print(f"Created mapping for {len(mapping)} classes.")
print("Person class:", mapping["1"])
