import sys

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

image = Image.open(sys.argv[1]).convert('RGB')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=float(sys.argv[2]))
print(len(processed))
results = processed[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
