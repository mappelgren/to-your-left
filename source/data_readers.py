import json
import random
from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
from torchvision.transforms._presets import ImageClassification


class ClevrDataset(Dataset):
    def __init__(self, scenes_json_file, image_path, max_number_samples) -> None:
        super().__init__()


        self.samples = []

        with open(scenes_json_file, 'r', encoding='utf-8') as f:
            scenes_metadata = json.load(f)
        for index, scene in enumerate(scenes_metadata['scenes']):
            if index == max_number_samples:
                break

            image = Image.open(image_path + scene['image_filename']).convert('RGB')

            bounding_boxes = self._get_bounding_boxes(image, scene)

            target_object = scene['groups']['target'][0]

            # target_x, target_y, _ = scene['objects'][target_object]['pixel_coords']
            # target_x = target_x / image.size[0]
            # target_y = target_y / image.size[1]

            # self.samples.append((bounding_boxes, torch.tensor((target_x, target_y)), bounding_boxes))

            target_bouding_box = bounding_boxes[target_object]
            del bounding_boxes[target_object]
            random.shuffle(bounding_boxes)

            sender_boxes = torch.stack([target_bouding_box, *bounding_boxes])
            target_index = random.randint(0, len(bounding_boxes))
            
            bounding_boxes.insert(target_index, target_bouding_box)
            receiver_boxes = torch.stack(bounding_boxes)
            self.samples.append((sender_boxes, target_index, receiver_boxes))
            


    def _get_bounding_boxes(self, image, scene):
        # preprocess = ResNet50_Weights.DEFAULT.transforms()
        BOUNDING_BOX_SIZE = image.size[0] / 5
        preprocess = partial(ImageClassification, crop_size=BOUNDING_BOX_SIZE)()

        object_bounding_boxes = []
        for obj in scene['objects']:
            x_center, y_center, _ = obj['pixel_coords']
            bounding_box = image.crop((
                x_center - BOUNDING_BOX_SIZE/2,
                y_center - BOUNDING_BOX_SIZE/2,
                x_center + BOUNDING_BOX_SIZE/2,
                y_center + BOUNDING_BOX_SIZE/2
                ))
            object_bounding_boxes.append(preprocess(bounding_box))

        object_bounding_boxes.extend([torch.zeros_like(object_bounding_boxes[0])] * (10 - len(object_bounding_boxes)))

        return object_bounding_boxes

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self) -> int:
        return len(self.samples)
    