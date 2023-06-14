import argparse
import json
import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet101_Weights, VGG19_Weights, resnet101, vgg19


class FeatureExtractor(ABC, nn.Module):
    @property
    @abstractmethod
    def feature_shape(self):
        ...


class DummyFeatureExtractor(FeatureExtractor):
    """Feature extractor that ouputs input unchanged

    Use this in a model if the input to the model are already extraced features (e.g. from .h5 files
    generated by this script) and not the images themselves.
    """

    @property
    def feature_shape(self):
        return (0,)

    def forward(self, data):
        return data


class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(
        self, pretrained=True, fine_tune=False, number_blocks=4, avgpool=True, fc=True
    ) -> None:
        super().__init__()

        if pretrained:
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet101()

        if number_blocks > 4 or number_blocks < 1:
            raise AttributeError("number of blocks need to be between 1 and 3")

        self.resnet = nn.Sequential(*list(resnet.children())[:5])
        self._feature_shape = (256, 56, 56)

        if number_blocks > 1:
            self.resnet.append(resnet.layer2)
            self._feature_shape = (512, 28, 28)

        if number_blocks > 2:
            self.resnet.append(resnet.layer3)
            self._feature_shape = (1024, 14, 14)

        if number_blocks > 3:
            self.resnet.append(resnet.layer4)
            self._feature_shape = (2048, 7, 7)

        if avgpool:
            self.resnet.append(resnet.avgpool)
            self._feature_shape = (2048, 1, 1)

        if fc:
            self.resnet.append(nn.Flatten())
            self.resnet.append(resnet.fc)
            self._feature_shape = (1000,)

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()

    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, data):
        return self.resnet(data)


class VggFeatureExtractor(FeatureExtractor):
    def __init__(
        self, pretrained=True, fine_tune=False, avgpool=True, classifier_layers=3
    ) -> None:
        super().__init__()

        if pretrained:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg19()

        self.vgg = nn.Sequential(vgg.features)
        self._feature_shape = (512, 7, 7)

        if avgpool:
            self.vgg.append(vgg.avgpool)
            self._feature_shape = (512, 7, 7)

        if classifier_layers < 0 or classifier_layers > 3:
            raise AttributeError(
                "number of classifier layers need to be between 0 and 3"
            )

        if classifier_layers > 0:
            self.vgg.append(nn.Flatten())
            self.vgg.append(vgg.classifier[0])
            self._feature_shape = (4096,)

        if classifier_layers > 1:
            self.vgg.append(vgg.classifier[1:4])
            self._feature_shape = (4096,)

        if classifier_layers > 2:
            self.vgg.append(vgg.classifier[4:7])
            self._feature_shape = (1000,)

        if not fine_tune:
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg.eval()

    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, data):
        return self.vgg(data)


def get_bounding_boxes(image, scene, preprocess, max_number_objects):
    BOUNDING_BOX_SIZE = image.size[0] / 5

    object_bounding_boxes = []
    for obj in scene["objects"]:
        x_center, y_center, _ = obj["pixel_coords"]
        bounding_box = image.crop(
            (
                x_center - BOUNDING_BOX_SIZE / 2,
                y_center - BOUNDING_BOX_SIZE / 2,
                x_center + BOUNDING_BOX_SIZE / 2,
                y_center + BOUNDING_BOX_SIZE / 2,
            )
        )
        object_bounding_boxes.append(preprocess(bounding_box))

    object_bounding_boxes.extend(
        [torch.zeros_like(object_bounding_boxes[0])]
        * (max_number_objects - len(object_bounding_boxes))
    )

    return object_bounding_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -- DATASET --
    parser.add_argument(
        "--image_dir", type=str, default=None, help="Path to the scene image dir"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument(
        "--bounding_boxes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set True, when features for each object in the image should be extracted",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default=None,
        help="Path to the scene dir (required when bounding boxes are extracted)",
    )
    # -- MODEL --
    parser.add_argument(
        "--feature_extractor",
        choices=["VGG", "ResNet"],
        help="model to load",
    )
    parser.add_argument(
        "--avgpool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include avgpool layer",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help="number of ResNet blocks in case of ResNet",
    )
    parser.add_argument(
        "--fc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include fc layer in case of ResNet",
    )
    parser.add_argument(
        "--classifier_layers",
        type=int,
        default=3,
        help="number of classifier layers",
    )

    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # -- SAVING --
    parser.add_argument(
        "--out_file",
        type=str,
        default="features.h5",
        help="directory, where the output should be saved",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        raise AttributeError("Device must be cpu or cuda")

    if args.feature_extractor == "VGG":
        feature_extractor = VggFeatureExtractor(
            avgpool=args.avgpool, classifier_layers=args.classifier_layers
        ).to(device)
        preprocess = VGG19_Weights.IMAGENET1K_V1.transforms()
    elif args.feature_extractor == "ResNet":
        feature_extractor = ResnetFeatureExtractor(
            number_blocks=args.num_blocks, avgpool=args.avgpool, fc=args.fc
        ).to(device)
        preprocess = ResNet101_Weights.IMAGENET1K_V2.transforms()
    else:
        raise AttributeError("no feature extractor specified")

    if args.bounding_boxes:
        scenes = sorted(os.listdir(args.scene_dir))

        # TODO optimize performance
        max_number_objects = 0
        for scene_file in scenes:
            with open(
                os.path.join(args.scene_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                scene = json.load(f)
                max_number_objects = max(len(scene["objects"]), max_number_objects)

        if args.batch_size % max_number_objects != 0:
            raise ValueError(
                f"batch_size must be divisible by the maximum  number of objects '{max_number_objects}' in an image"
            )

        shape = [len(scenes), max_number_objects, *feature_extractor.feature_shape]

        with h5py.File(args.out_file, "w") as f:
            feature_dataset = f.create_dataset("features", shape, dtype=np.float32)
            feature_dataset.attrs["image_size"] = Image.open(
                os.path.join(args.image_dir, os.listdir(args.image_dir)[0])
            ).size

            batch = []
            i_start = 0
            for image_index, scene_file in enumerate(scenes):
                if image_index % args.batch_size == 0:
                    print(f"processing image {image_index}...", end="\r")

                with open(
                    os.path.join(args.scene_dir, scene_file), "r", encoding="utf-8"
                ) as s:
                    scene = json.load(s)

                image = Image.open(
                    os.path.join(args.image_dir, scene["image_filename"])
                ).convert("RGB")

                bounding_boxes = get_bounding_boxes(
                    image, scene, preprocess, max_number_objects
                )

                batch.extend(
                    [bounding_box.to(device) for bounding_box in bounding_boxes]
                )

                if len(batch) == args.batch_size or image_index == len(scenes) - 1:
                    number_images = int(len(batch) / max_number_objects)

                    with torch.no_grad():
                        features = feature_extractor(torch.stack(batch)).cpu()

                    features_by_image = features.view(
                        number_images, max_number_objects, -1
                    )
                    i_end = i_start + number_images
                    feature_dataset[i_start:i_end] = features_by_image

                    i_start = i_end
                    batch = []
            print()

    else:
        images = sorted(os.listdir(args.image_dir))
        shape = [len(images), *feature_extractor.feature_shape]

        with h5py.File(args.out_file, "w") as f:
            feature_dataset = f.create_dataset("features", shape, dtype=np.float32)
            feature_dataset.attrs["image_size"] = Image.open(
                os.path.join(args.image_dir, images[0])
            ).size
            batch = []
            i_start = 0
            for image_index, image_file in enumerate(images):
                if image_index % args.batch_size == 0:
                    print(f"processing image {image_index}...", end="\r")

                image = Image.open(os.path.join(args.image_dir, image_file)).convert(
                    "RGB"
                )
                preprocessed_image = preprocess(image).to(device)
                batch.append(preprocessed_image)

                if len(batch) == args.batch_size or image_index == len(images) - 1:
                    with torch.no_grad():
                        features = feature_extractor(torch.stack(batch)).cpu()
                    i_end = i_start + len(batch)
                    feature_dataset[i_start:i_end] = features

                    i_start = i_end
                    batch = []
            print()
