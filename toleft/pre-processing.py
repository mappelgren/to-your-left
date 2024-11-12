from feature_extractors import ResnetFeatureExtractor
import torch
from torchvision.models import ResNet101_Weights, resnet101
import os
import h5py
from PIL import Image
import numpy as np


def pre_process_rotation_dataset(image_dir, output_name = 'rot_dataset.h5py', batch_size=32, device='cuda'):
    device = torch.device(device)
    feature_extractor = ResnetFeatureExtractor(pretrained=True, fine_tune=False, number_blocks=3, avgpool=False,
                                               fc=False).to(device)
    preprocess = ResNet101_Weights.IMAGENET1K_V2.transforms()

    images = sorted(os.listdir(image_dir + 'rot0'))
    shape = [len(images), *feature_extractor.feature_shape]

    with h5py.File(output_name, "w") as f:
        rot0_dataset = f.create_dataset("rot0", shape, dtype=np.float32)
        rot90_dataset = f.create_dataset("rot90", shape, dtype=np.float32)
        rot180_dataset = f.create_dataset("rot180", shape, dtype=np.float32)
        rot270_dataset = f.create_dataset("rot270", shape, dtype=np.float32)

        datasets = {'rot0': rot0_dataset, 'rot90': rot90_dataset, 'rot180': rot180_dataset, 'rot270': rot270_dataset}
        img_size = Image.open(
            os.path.join(image_dir, 'rot0', images[0])
        ).size
        for d in datasets.values():
            d.attrs["image_size"] = img_size

        for rotation, dataset in datasets.items():
            images = sorted(os.listdir(image_dir + rotation))
            batch = []
            i_start = 0
            for image_index, image_file in enumerate(images):
                if image_index % batch_size == 0:
                    print(f"processing image {image_index}...", end="\r")

                image = Image.open(os.path.join(image_dir, rotation, image_file)).convert(
                    "RGB"
                )
                preprocessed_image = preprocess(image).to(device)
                batch.append(preprocessed_image)

                if len(batch) == batch_size or image_index == len(images) - 1:
                    with torch.no_grad():
                        # pylint: disable-next=not-callable
                        features = feature_extractor(torch.stack(batch)).cpu()
                    i_end = i_start + len(batch)
                    dataset[i_start:i_end] = features

                    i_start = i_end
                    batch = []
        print()