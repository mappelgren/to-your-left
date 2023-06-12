import itertools
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import h5py
import torch
from mlt.preexperiments.data_readers import CaptionGeneratorDataset
from mlt.preexperiments.feature_extractors import (
    DummyFeatureExtractor,
    FeatureExtractor,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet101_Weights


@dataclass
class ReferentialGameSample:
    image_1_id: str
    image_2_id: str
    image_1: torch.Tensor
    image_2: torch.Tensor

    # target
    target_image: torch.Tensor


class _BatchIterator(Iterator):
    def __init__(self, loader, batch_size, n_batches, seed) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batches_generated = 0
        random.seed(seed)

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        concept_dict = self.loader.dataset.concept_dict

        sender_inputs = []
        targets = []
        receiver_inputs = []

        for _ in range(self.batch_size):
            concepts = random.sample(concept_dict.keys(), 2)

            image_1 = random.choice(concept_dict[concepts[0]])
            image_2 = random.choice(concept_dict[concepts[1]])

            sender_input = torch.stack((image_1, image_2))

            target_image = torch.tensor(random.randint(0, 1))

            receiver_input = [image_2]
            receiver_input.insert(target_image, image_1)
            receiver_input = torch.stack(receiver_input)

            sender_inputs.append(sender_input)
            targets.append(target_image)
            receiver_inputs.append(receiver_input)

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
        )


class LazaridouReferentialGameLoader(DataLoader):
    def __init__(self, batch_size, batches_per_epoch, *args, seed=None, **kwargs):
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed

        # batch_size is part of standard DataLoader arguments
        kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = self.seed
        return _BatchIterator(
            self,
            batch_size=self.batch_size,
            n_batches=self.batches_per_epoch,
            seed=seed,
        )


class LazaridouReferentialGameDataset(Dataset):
    def __init__(self, data_root_path) -> None:
        super().__init__()

        feature_file_path = os.path.join(
            data_root_path, "train", "ours_images_single_sm0.h5"
        )
        label_file_path = os.path.join(
            data_root_path, "train", "ours_images_single_sm0.objects"
        )

        fc = h5py.File(feature_file_path, "r")

        data = torch.FloatTensor(list(fc["dataset_1"]))

        # normalise data
        img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        self.images = data / img_norm

        with open(label_file_path, "rb") as f:
            labels = pickle.load(f)

        self.concept_dict = defaultdict(list)
        for data, label in zip(self.images, labels):
            self.concept_dict[label].append(data)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self) -> int:
        return len(self.images)


@dataclass
class BoundingBoxReferentialGameSample:
    bounding_boxes: list[torch.Tensor]
    target_index: int
    target_order: list[int]
    image_id: str


class DaleTwoReferentialGameDataset(Dataset):
    def __init__(
        self,
        data_root_path,
        feature_extractor: FeatureExtractor,
        max_number_samples=100,
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()

        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        self.samples: list[BoundingBoxReferentialGameSample] = []

        scenes_json_dir = os.path.join(data_root_path, "scenes/")
        image_path = os.path.join(data_root_path, "images/")

        scenes = os.listdir(scenes_json_dir)
        print("loading scenes...")
        loaded_scenes = []
        max_number_objects = 0
        for scene_file in scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                loaded_scenes.append(json.load(f))
                max_number_objects = max(
                    len(loaded_scenes[-1]["objects"]), max_number_objects
                )

        if max_number_samples > -1:
            selected_scenes = random.sample(loaded_scenes, max_number_samples)
        else:
            selected_scenes = loaded_scenes

        for scene_index, scene in enumerate(selected_scenes):
            if scene_index % 100 == 0:
                print(f"processing scene {scene_index}...", end="\r")

            image = Image.open(image_path + scene["image_filename"]).convert("RGB")
            bounding_boxes = []
            for object_index in range(len(scene["objects"])):
                bounding_box = preprocess(
                    self._get_bounding_box(image, scene, object_index)
                )
                with torch.no_grad():
                    bounding_box = (
                        feature_extractor(bounding_box.to(device).unsqueeze(dim=0))
                        .squeeze(dim=0)
                        .cpu()
                    )
                bounding_boxes.append(bounding_box)

            target_object = scene["groups"]["target"][0]

            # move target bounding box to first position
            bounding_boxes = [
                bounding_boxes[target_object],
                *[
                    box
                    for index, box in enumerate(bounding_boxes)
                    if index != target_object
                ],
            ]

            bounding_boxes.extend(
                [torch.zeros_like(bounding_boxes[0])]
                * (max_number_objects - len(bounding_boxes))
            )

            indices = list(range(len(bounding_boxes)))
            random.shuffle(indices)

            target_index = indices.index(target_object)

            self.samples.append(
                BoundingBoxReferentialGameSample(
                    target_index=target_index,
                    bounding_boxes=bounding_boxes,
                    target_order=indices,
                    image_id=scene["image_filename"].removesuffix(".png"),
                )
            )
        print()

    def _get_bounding_box(self, image, scene, target_index):
        BOUNDING_BOX_SIZE = image.size[0] / 5

        x_center, y_center, _ = scene["objects"][target_index]["pixel_coords"]
        bounding_box = image.crop(
            (
                x_center - BOUNDING_BOX_SIZE / 2,
                y_center - BOUNDING_BOX_SIZE / 2,
                x_center + BOUNDING_BOX_SIZE / 2,
                y_center + BOUNDING_BOX_SIZE / 2,
            )
        )

        return bounding_box

    def __getitem__(self, index):
        sample = self.samples[index]

        sender_input = torch.stack(sample.bounding_boxes)

        target_index = torch.tensor(sample.target_index)

        receiver_input = torch.stack(
            [sample.bounding_boxes[i] for i in sample.target_order]
        )

        return (sender_input, target_index, receiver_input)

    def __len__(self) -> int:
        return len(self.samples)


class BoundingBoxReferentialGameDataset(Dataset):
    def __init__(
        self,
        scenes_json_dir,
        image_path="",
        max_number_samples=0,
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
    ) -> None:
        super().__init__()

        self.samples: list[ReferentialGameSample] = []

        scenes = os.listdir(scenes_json_dir)
        print("loading scenes...")
        loaded_scenes = []
        for scene_file in scenes:
            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                loaded_scenes.append(json.load(f))

        print("creating combinations...")
        scene_combinations = list(
            itertools.product(range(len(loaded_scenes)), repeat=2)
        )
        print("shuffling...")
        random.shuffle(scene_combinations)

        print("looping...")

        for scene_1_index, scene_2_index in scene_combinations:
            if len(self.samples) == max_number_samples:
                break

            scene_1 = loaded_scenes[scene_1_index]
            scene_2 = loaded_scenes[scene_2_index]

            target_object_1_index = scene_1["groups"]["target"][0]
            target_object_2_index = scene_2["groups"]["target"][0]

            target_object_1 = scene_1["objects"][target_object_1_index]
            target_object_2 = scene_2["objects"][target_object_2_index]

            # no matching attribute
            if (
                target_object_1["color"] == target_object_2["color"]
                or target_object_1["shape"] == target_object_2["shape"]
                or target_object_1["size"] == target_object_2["size"]
            ):
                continue

            image_1 = Image.open(image_path + scene_1["image_filename"]).convert("RGB")
            image_2 = Image.open(image_path + scene_2["image_filename"]).convert("RGB")
            target_image = random.randint(0, 1)

            self.samples.append(
                ReferentialGameSample(
                    image_1_id=scene_1["image_filename"].removesuffix(".png"),
                    image_2_id=scene_2["image_filename"].removesuffix(".png"),
                    image_1=preprocess(
                        self._get_bounding_box(image_1, scene_1, target_object_1_index)
                    ),
                    image_2=preprocess(
                        self._get_bounding_box(image_2, scene_2, target_object_2_index)
                    ),
                    target_image=torch.tensor(target_image),
                )
            )

    def _get_bounding_box(self, image, scene, target_index):
        BOUNDING_BOX_SIZE = image.size[0] / 5

        x_center, y_center, _ = scene["objects"][target_index]["pixel_coords"]
        bounding_box = image.crop(
            (
                x_center - BOUNDING_BOX_SIZE / 2,
                y_center - BOUNDING_BOX_SIZE / 2,
                x_center + BOUNDING_BOX_SIZE / 2,
                y_center + BOUNDING_BOX_SIZE / 2,
            )
        )

        return bounding_box

    def __getitem__(self, index):
        sample = self.samples[index]
        sender_input = torch.stack((sample.image_1, sample.image_2))

        receiver_input = [sample.image_2]
        receiver_input.insert(sample.target_image, sample.image_1)
        receiver_input = torch.stack(receiver_input)

        target = sample.target_image
        return (sender_input, target, receiver_input)

    def __len__(self) -> int:
        return len(self.samples)


class GameCaptionGeneratorDataset(CaptionGeneratorDataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        sender_input = torch.stack((sample.image, sample.masked_image))

        receiver_input = sample.image

        target = sample.caption[1:]

        return (sender_input, target, receiver_input)
