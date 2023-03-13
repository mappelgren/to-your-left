import sys

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

file_name = sys.argv[1]
threshold = float(sys.argv[2])

image = Image.open(file_name).convert('RGB')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
results = processed[0]
print(f'Threshold: {threshold}')
print(f'number of boxes: {len(results['scores'])}')
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score < threshold:
        continue
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 6)} at location {box}"
    )
