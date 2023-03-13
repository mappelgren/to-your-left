import json
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights


@dataclass
class Sample:
    image_id: str
    image: torch.Tensor
    number_objects: int

class NumberObjectsDataset(Dataset):
    def __init__(self, scenes_json_file, image_path, max_number_samples) -> None:
        super().__init__()
        transform = transforms.ToTensor()
        resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples = []
        self.max_number = 0
        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for index, scene in enumerate(scenes_metadata['scenes']):
            if index == max_number_samples:
                break

            number_objects = len(scene['objects'])
            self.max_number = max(self.max_number, number_objects)
            # image = Image.open(image_path + scene['image_filename']).resize((100,100)).convert('RGB')
            # self.samples.append((transform(image), number_objects))
            
            image = Image.open(image_path + scene['image_filename']).convert('RGB')
            transformed = resnet_transform(image)
            self.samples.append((transformed, number_objects, transformed))
            # self.samples.append(Sample(image_id=scene['image_filename'].removesuffix('.png'),
            #                            image=transform(image),
            #                            number_objects=number_objects))

    def __getitem__(self, index) -> Sample:
        return self.samples[index]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_number_labels(self) -> int:
        return self.max_number + 1
