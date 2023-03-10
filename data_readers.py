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
    def __init__(self, scenes_json_file, image_path) -> None:
        super().__init__()
        transform = transforms.ToTensor()

        self.samples = []
        self.max_number = 0
        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for scene in scenes_metadata['scenes']:
            number_objects = len(scene['objects'])
            self.max_number = max(self.max_number, number_objects)
            # image = Image.open(image_path + scene['image_filename']).resize((100,100)).convert('RGB')
            # self.samples.append((transform(image), number_objects))
            
            image = Image.open(image_path + scene['image_filename'])
            self.samples.append((ResNet18_Weights.IMAGENET1K_V1.transforms(image), number_objects))
            # self.samples.append(Sample(image_id=scene['image_filename'].removesuffix('.png'),
            #                            image=transform(image),
            #                            number_objects=number_objects))

    def __getitem__(self, index) -> Sample:
        return self.samples[index]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_number_labels(self) -> int:
        return self.max_number + 1
