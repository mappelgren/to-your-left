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
    OneHotGeneratorDataset,
    OneHotGeneratorSample,
)
from mlt.util import Persistor, load_tensor
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
    def __init__(self, images, concept_dict) -> None:
        super().__init__()

        self.images = images
        self.concept_dict = concept_dict

    @classmethod
    def load(cls, data_root_dir, *_args, **_kwargs):
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
        images = data / img_norm

        with open(label_file_path, "rb") as f:
            labels = pickle.load(f)

        concept_dict = defaultdict(list)
        for data, label in zip(images, labels):
            concept_dict[label].append(data)

        return cls(images, concept_dict)

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
        file_path: str,
    ) -> None:
        super().__init__()

        self.file = file_path

        with h5py.File(file_path, "r") as f:
            self.num_samples = len(list(f.values())[0])

    @classmethod
    def load(
        cls,
        scenes_json_dir,
        bounding_box_loader: ImageLoader,
        persistor: Persistor,
        *_args,
        max_number_samples=100,
        **_kwargs,
    ):
        samples: list[DaleReferentialGameSample] = []

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
            _, bounding_boxes, _ = bounding_box_loader.get_image(image_id)

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

            samples.append(
                DaleReferentialGameSample(
                    target_index=target_index,
                    bounding_boxes=bounding_boxes,
                    target_order=indices,
                    image_id=scene["image_filename"].removesuffix(".png"),
                )
            )
        print()

        persistor.save(samples)
        return cls(persistor.file_path)

    def __getitem__(self, index):
        with h5py.File(self.file, "r") as f:
            return DaleReferentialGameSample(
                image_id=str(f["image_id"][index], "utf-8"),
                target_index=load_tensor(f["target_index"][index]),
                bounding_boxes=load_tensor(f["bounding_boxes"][index]),
                target_order=load_tensor(f["target_order"][index]),
            )

    def __len__(self) -> int:
        return self.num_samples


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
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.bounding_boxes)
            targets.append(sample.target_index)

            receiver_inputs.append(
                torch.stack([sample.bounding_boxes[i] for i in sample.target_order])
            )

            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "image_id": torch.tensor(image_ids),
            },
        )


class OneHotGeneratorGameDataset(OneHotGeneratorDataset):
    def __getitem__(self, index):
        with h5py.File(self.file, "r") as f:
            return OneHotGeneratorSample(
                image_id=str(f["image_id"][index], "utf-8"),
                image=load_tensor(f["image"][index]),
                attribute_encoding=load_tensor(f["attribute_encoding"][index]),
                target_encoding=load_tensor(f["target_encoding"][index]),
                non_target_encodings=load_tensor(f["non_target_encodings"][index]),
                bounding_boxes=load_tensor(f["bounding_boxes"][index]),
            )


class OneHotGeneratorGameBatchIterator(GameBatchIterator):
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
        samples: list[OneHotGeneratorSample] = [
            self.loader.dataset[i] for i in sampled_indices
        ]

        sender_inputs = []
        targets = []
        receiver_inputs = []
        non_target_encodings = []
        train_modes = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.bounding_boxes)
            targets.append(sample.target_encoding)

            receiver_inputs.append(sample.image)
            non_target_encodings.append(sample.non_target_encodings)
            train_modes.append(self.train_mode)
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "non_target_encodings": torch.stack(non_target_encodings),
                "train_mode": torch.tensor(train_modes),
                "image_id": torch.tensor(image_ids),
            },
        )


class CaptionGeneratorGameDataset(CaptionGeneratorDataset):
    def __getitem__(self, index):
        with h5py.File(self.file, "r") as f:
            return CaptionGeneratorSample(
                image_id=str(f["image_id"][index], "utf-8"),
                image=load_tensor(f["image"][index]),
                caption=load_tensor(f["caption"][index]),
                masked_image=load_tensor(f["masked_image"][index]),
                non_target_captions=load_tensor(f["non_target_captions"][index]),
                bounding_boxes=load_tensor(f["bounding_boxes"][index]),
            )


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
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "masked_image": torch.stack(masked_images),
                "caption": torch.stack(captions),
                "non_target_captions": torch.stack(non_target_captions),
                "train_mode": torch.tensor(train_modes),
                "image_id": torch.tensor(image_ids),
            },
        )


class BoundingBoxCaptionGeneratorGameBatchIterator(GameBatchIterator):
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
        captions = []
        non_target_captions = []
        train_modes = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.bounding_boxes)
            targets.append(sample.caption[1:])

            receiver_inputs.append(sample.image)
            captions.append(sample.caption)
            non_target_captions.append(sample.non_target_captions)
            train_modes.append(self.train_mode)
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "caption": torch.stack(captions),
                "non_target_captions": torch.stack(non_target_captions),
                "train_mode": torch.tensor(train_modes),
                "image_id": torch.tensor(image_ids),
            },
        )


class CoordinatePredictorGameDataset(CoordinatePredictorDataset):
    def __getitem__(self, index):
        with h5py.File(self.file, "r") as f:
            return CoordinatePredictorSample(
                image_id=str(f["image_id"][index], "utf-8"),
                image=load_tensor(f["image"][index]),
                target_pixels=load_tensor(f["target_pixels"][index]),
                target_region=load_tensor(f["target_region"][index]),
                attribute_tensor=load_tensor(f["attribute_tensor"][index]),
                locations=load_tensor(f["locations"][index]),
                masked_image=load_tensor(f["masked_image"][index]),
                bounding_boxes=load_tensor(f["bounding_boxes"][index]),
            )


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
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "masked_image": torch.stack(masked_images),
                "attribute_tensor": torch.stack(attibute_tensors),
                "image_id": torch.tensor(image_ids),
            },
        )


class BoundingBoxCoordinatePredictorGameBatchIterator(GameBatchIterator):
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
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.bounding_boxes)
            targets.append(sample.target_pixels)

            receiver_inputs.append(sample.image)
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(targets),
            torch.stack(receiver_inputs),
            {
                "image_id": torch.tensor(image_ids),
            },
        )


class AttentionPredictorGameBatchIterator(GameBatchIterator):
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
        target_regions = []
        receiver_inputs = []
        masked_images = []
        attibute_tensors = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.image)
            target_regions.append(sample.target_region)

            receiver_inputs.append(sample.image)
            masked_images.append(sample.masked_image)
            attibute_tensors.append(sample.attribute_tensor)
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(target_regions),
            torch.stack(receiver_inputs),
            {
                "masked_image": torch.stack(masked_images),
                "attribute_tensor": torch.stack(attibute_tensors),
                "image_id": torch.tensor(image_ids),
            },
        )


class BoundingBoxAttentionPredictorGameBatchIterator(GameBatchIterator):
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
        target_regions = []
        receiver_inputs = []
        image_ids = []

        for sample in samples:
            sender_inputs.append(sample.bounding_boxes)
            target_regions.append(sample.target_region)

            receiver_inputs.append(sample.image)
            image_ids.append(int(sample.image_id[-6:]))

        return (
            torch.stack(sender_inputs),
            torch.stack(target_regions),
            torch.stack(receiver_inputs),
            {
                "image_id": torch.tensor(image_ids),
            },
        )
