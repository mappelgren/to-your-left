import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import h5py
import torch
from mlt.image_loader import ImageLoader
from mlt.preexperiments.data_readers import (
    CaptionGeneratorDataset,
    CaptionGeneratorSample,
    CoordinatePredictorDataset,
    CoordinatePredictorSample,
)
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class ReferentialGameSample:
    image_1_id: str
    image_2_id: str
    image_1: torch.Tensor
    image_2: torch.Tensor

    # target
    target_image: torch.Tensor


class GameBatchIterator(Iterator, ABC):
    @abstractmethod
    def __init__(self, loader, batch_size, n_batches, train_mode, seed):
        ...


class GameLoader(DataLoader):
    def __init__(
        self,
        iterator: GameBatchIterator,
        batch_size,
        batches_per_epoch,
        train_mode,
        *args,
        seed=None,
        **kwargs,
    ):
        self.iterator = iterator
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.train_mode = train_mode

        # batch_size is part of standard DataLoader arguments
        kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = self.seed
        return self.iterator(
            self,
            batch_size=self.batch_size,
            n_batches=self.batches_per_epoch,
            train_mode=self.train_mode,
            seed=seed,
        )


class LazaridouReferentialGameDataset(Dataset):
    def __init__(self, data_root_dir, *_args, **_kwargs) -> None:
        super().__init__()

        feature_file_path = os.path.join(
            data_root_dir, "train", "ours_images_single_sm0.h5"
        )
        label_file_path = os.path.join(
            data_root_dir, "train", "ours_images_single_sm0.objects"
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


class LazaridouReferentialGameBatchIterator(GameBatchIterator):
    def __init__(self, loader, batch_size, n_batches, train_mode, seed) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batches_generated = 0
        self.train_mode = train_mode
        self.random_seed = random.Random(seed)

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        if isinstance(self.loader.dataset, Subset):
            concept_dict = self.loader.dataset.dataset.concept_dict
        else:
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


@dataclass
class DaleReferentialGameSample:
    bounding_boxes: list[torch.Tensor]
    target_index: int
    target_order: list[int]
    image_id: str


class DaleReferentialGameDataset(Dataset):
    def __init__(
        self,
        scenes_json_dir,
        image_loader: ImageLoader,
        *_args,
        max_number_samples=100,
        **_kwargs,
    ) -> None:
        super().__init__()

        self.samples: list[DaleReferentialGameSample] = []

        scenes = os.listdir(scenes_json_dir)
        print("sampling scenes...")
        if max_number_samples > -1:
            selected_scenes = random.sample(scenes, max_number_samples)
        else:
            selected_scenes = scenes

        for scene_index, scene_file in enumerate(selected_scenes):
            if scene_index % 50 == 0:
                print(f"processing scene {scene_index}...", end="\r")

            with open(
                os.path.join(scenes_json_dir, scene_file), "r", encoding="utf-8"
            ) as f:
                scene = json.load(f)

            image_id = scene_file.removesuffix(".json")
            _, bounding_boxes, _ = image_loader.get_image(image_id)

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

            indices = list(range(len(bounding_boxes)))
            random.shuffle(indices)

            target_index = indices.index(target_object)

            self.samples.append(
                DaleReferentialGameSample(
                    target_index=target_index,
                    bounding_boxes=bounding_boxes,
                    target_order=indices,
                    image_id=scene["image_filename"].removesuffix(".png"),
                )
            )
        print()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class DaleReferentialGameBatchIterator(GameBatchIterator):
    def __init__(self, loader, batch_size, n_batches, train_mode, seed) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batches_generated = 0
        self.train_mode = train_mode
        self.random_seed = random.Random(seed)

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        sampled_indices = self.random_seed.sample(
            range(len(self.loader.dataset)), self.batch_size
        )
        samples: list[DaleReferentialGameSample] = [
            self.loader.dataset[i] for i in sampled_indices
        ]

        sender_inputs = []
        targets = []
        receiver_inputs = []

        for sample in samples:
            sender_inputs.append(torch.stack(sample.bounding_boxes))
            targets.append(torch.tensor(sample.target_index))

            receiver_inputs.append(
                torch.stack([sample.bounding_boxes[i] for i in sample.target_order])
            )

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
        )


class CaptionGeneratorGameDataset(CaptionGeneratorDataset):
    def __getitem__(self, index):
        return self.samples[index]


class CaptionGeneratorGameBatchIterator(GameBatchIterator):
    def __init__(self, loader, batch_size, n_batches, train_mode, seed) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batches_generated = 0
        self.train_mode = train_mode
        self.random_seed = random.Random(seed)

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        sampled_indices = self.random_seed.sample(
            range(len(self.loader.dataset)), self.batch_size
        )
        samples: list[CaptionGeneratorSample] = [
            self.loader.dataset[i] for i in sampled_indices
        ]

        sender_inputs = []
        targets = []
        receiver_inputs = []
        masked_images = []
        captions = []
        non_target_captions = []
        train_modes = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.image)
            targets.append(sample.caption[1:])

            receiver_inputs.append(sample.image)
            masked_images.append(sample.masked_image)
            captions.append(sample.caption)
            non_target_captions.append(sample.non_target_captions)
            train_modes.append(self.train_mode)
            image_ids.append(sample.image_id)

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "masked_image": torch.stack(masked_images),
                "caption": torch.stack(captions),
                "non_target_captions": torch.stack(non_target_captions),
                "train_mode": torch.tensor(train_modes),
                "image_id": torch.stack(image_ids),
            },
        )


class CoordinatePredictorGameDataset(CoordinatePredictorDataset):
    def __getitem__(self, index):
        return self.samples[index]


class CoordinatePredictorGameBatchIterator(GameBatchIterator):
    def __init__(self, loader, batch_size, n_batches, train_mode, seed) -> None:
        self.loader = loader
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batches_generated = 0
        self.train_mode = train_mode
        self.random_seed = random.Random(seed)

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        sampled_indices = self.random_seed.sample(
            range(len(self.loader.dataset)), self.batch_size
        )
        samples: list[CoordinatePredictorSample] = [
            self.loader.dataset[i] for i in sampled_indices
        ]

        sender_inputs = []
        targets = []
        receiver_inputs = []
        masked_images = []
        attibute_tensors = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.image)
            targets.append(sample.target_pixels)

            receiver_inputs.append(sample.image)
            masked_images.append(sample.masked_image)
            attibute_tensors.append(sample.attribute_tensor)
            image_ids.append(sample.image_id)

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "masked_image": torch.stack(masked_images),
                "attribute_tensor": torch.stack(attibute_tensors),
                "image_id": torch.stack(image_ids),
            },
        )
