import argparse
from dataclasses import dataclass
from test import (
    BoundingBoxClassifierTester,
    CaptionGeneratorTester,
    CoordinatePredictorTester,
    Tester,
)
from typing import Callable

import torch
from data_readers import (
    BoundingBoxClassifierDataset,
    CaptionGeneratorDataset,
    CoordinatePredictorDataset,
)
from models import (
    AttributeCoordinatePredictor,
    AttributeLocationCoordinatePredictor,
    BoundingBoxClassifier,
    CaptionDecoder,
    CaptionGenerator,
    CoordinatePredictor,
    ImageEncoder,
)
from save import (
    CaptionOutputProcessor,
    ModelSaver,
    PixelOutputProcessor,
    StandardOutputProcessor,
)
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torcheval.metrics import Mean


def pixel_loss(model_output, ground_truth):
    loss = torch.diagonal(torch.cdist(model_output, ground_truth.float()))

    return torch.mean(loss)


@dataclass
class ModelDefinition:
    dataset: Dataset
    dataset_args: dict
    model: Module
    model_args: dict
    loss_function: Callable
    tester: Tester
    output_processor: StandardOutputProcessor
    output_processor_args: dict


models = {
    "coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={},
        model=CoordinatePredictor,
        model_args={},
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={"output_fields": ("image_id", "x", "y")},
    ),
    "attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"encode_attributes": True},
        model=AttributeCoordinatePredictor,
        model_args={},
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={"output_fields": ("image_id", "x", "y")},
    ),
    "attribute_location_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"encode_attributes": True, "encode_locations": True},
        model=AttributeLocationCoordinatePredictor,
        model_args={},
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={"output_fields": ("image_id", "x", "y")},
    ),
    "bounding_box_classifier": ModelDefinition(
        dataset=BoundingBoxClassifierDataset,
        dataset_args={},
        model=BoundingBoxClassifier,
        model_args={},
        loss_function=nn.CrossEntropyLoss(),
        tester=BoundingBoxClassifierTester,
        output_processor=StandardOutputProcessor,
        output_processor_args={"output_fields": ("image_id", "bounding_box")},
    ),
    "caption_generator": ModelDefinition(
        dataset=CaptionGeneratorDataset,
        dataset_args={},
        model=CaptionGenerator,
        model_args={
            "image_encoder": ImageEncoder(2048),
            "caption_decoder": CaptionDecoder(14, 128, 2048),
            "encoded_sos": 0,
        },
        loss_function=nn.CrossEntropyLoss(),
        tester=CaptionGeneratorTester,
        output_processor=CaptionOutputProcessor,
        output_processor_args={"output_fields": ("image_id", "caption")},
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -- DATASET --
    parser.add_argument(
        "--scene_json_dir", type=str, default=None, help="Path to the scene json dir"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None, help="Path to the scene image dir"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="max samples to load"
    )

    # -- MODEL --
    parser.add_argument(
        "--model",
        choices=models.keys(),
        help="model to load",
    )

    # -- TRAINING --
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    # -- SAVING --
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/",
        help="directory, where the output should be saved",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        raise AttributeError("Device must be cpu or cuda")

    dataset = models[args.model].dataset(
        args.scene_json_dir,
        args.image_dir,
        args.max_samples,
        **models[args.model].dataset_args,
    )

    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(
        dataset, (train_dataset_length, test_dataset_length)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    output_processor = models[args.model].output_processor(
        dataset=dataset, **models[args.model].output_processor_args
    )
    model_saver = ModelSaver(args.out_dir, args.model, output_processor)
    tester = models[args.model].tester()

    model = models[args.model].model(**models[args.model].model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_function = models[args.model].loss_function
    test_function = models[args.model].tester

    log = [str(args) + "\n"]
    print(f"Batches per epoch: {len(train_loader)}")
    for epoch in range(args.epochs):
        total_loss = Mean(device=device)
        model.train()
        train_outputs = []
        for i, (model_input, ground_truth, image_id) in enumerate(train_loader):
            if isinstance(model_input, list):
                model_input = [t.to(device) for t in model_input]
            else:
                model_input.to(device)
            ground_truth = ground_truth.to(device)

            output = model(model_input)
            train_outputs.extend(zip(image_id, output.detach()))

            loss = loss_function(output, ground_truth)

            total_loss.update(loss)

            loss_string = f"epoch {epoch}, batch {i}: {total_loss.compute():.4f}"
            print(
                loss_string,
                end="\r",
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()
        metrics, test_outputs = tester.test(model, test_loader, device)
        print(metrics)
        log.append(loss_string + "\n")
        log.append(str(metrics) + "\n")

    model_saver.save_model(model, f"{model.__class__.__name__}.pth")
    model_saver.save_log(log, "log.txt")
    model_saver.save_output(test_outputs, "test_outputs.csv")
    model_saver.save_output(train_outputs, "train_outputs.csv")
