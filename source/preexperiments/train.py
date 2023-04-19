import argparse
import os
from dataclasses import dataclass
from time import gmtime, strftime
from typing import Callable

import torch
from classification_models import (
    AttributeCoordinatePredictor,
    AttributeLocationCoordinatePredictor,
    BoundingBoxClassifier,
    CoordinatePredictor,
)
from data_readers import BoundingBoxClassifierDataset, CoordinatePredictorDataset
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torcheval.metrics import BinaryAccuracy, Mean, MulticlassAccuracy


def save_model(directory, model_name, model, log, train_output, test_output):
    folder_name = f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}_{model_name}'
    sub_folder = os.path.join(directory, folder_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    save_to_csv(train_output, os.path.join(sub_folder, "train_outputs.csv"))
    save_to_csv(test_output, os.path.join(sub_folder, "test_outputs.csv"))

    with open(os.path.join(sub_folder, "log.txt"), "w", encoding="utf-8") as f:
        f.writelines(log)

    torch.save(model.state_dict(), os.path.join(sub_folder, "model.pth"))


def save_to_csv(data, file):
    with open(file, "w", encoding="utf-8") as f:
        f.write("image_id,x,y\n")
        for image_id, pixels in data:
            f.write(f"{image_id},{pixels[0]},{pixels[1]}\n")


def pixel_loss(model_output, ground_truth):
    loss = torch.diagonal(torch.cdist(model_output, ground_truth.float()))

    return torch.mean(loss)


def test_coordinate_predictor(model, test_loader):
    model.eval()
    accuracy = BinaryAccuracy(device=device)
    mean = Mean(device=device)

    test_outputs = []
    for model_input, ground_truth, image_id in test_loader:
        model_input = [t.to(device) for t in model_input]
        ground_truth = ground_truth.to(device)

        output = model(model_input).detach()
        test_outputs.extend(zip(image_id, output))

        distances = torch.diagonal(torch.cdist(output, ground_truth.float()))
        mean.update(distances)

        positives = torch.where(distances < 20, distances, 0)
        accuracy.update(positives, torch.ones_like(positives))

    return {
        "accuracy": f"{accuracy.compute():.2f}",
        "mean test loss": f"{mean.compute():.2f}",
    }, test_outputs


def test_bounding_box_classifier(model, test_loader):
    model.eval()
    accuracy = MulticlassAccuracy()

    test_outputs = []
    for model_input, ground_truth, image_id in test_loader:
        model_input = model_input.to(device)
        ground_truth = ground_truth.to(device)
        output = model(model_input).detach()
        max_indices = torch.max(output, dim=1)[1]
        test_outputs.extend(zip(image_id, max_indices))

        accuracy.update(max_indices, ground_truth)

    return {
        "accuracy": f"{accuracy.compute():.2f}",
    }, test_outputs


@dataclass
class ModelDefinition:
    dataset: Dataset
    dataset_args: dict
    model: Module
    loss_function: Callable
    test_function: Callable


models = {
    "coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={},
        model=CoordinatePredictor,
        loss_function=pixel_loss,
        test_function=test_coordinate_predictor,
    ),
    "attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"encode_attributes": True},
        model=AttributeCoordinatePredictor,
        loss_function=pixel_loss,
        test_function=test_coordinate_predictor,
    ),
    "attribute_location_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"encode_attributes": True, "encode_locations": True},
        model=AttributeLocationCoordinatePredictor,
        loss_function=pixel_loss,
        test_function=test_coordinate_predictor,
    ),
    "bounding_box_classifier": ModelDefinition(
        dataset=BoundingBoxClassifierDataset,
        dataset_args={},
        model=BoundingBoxClassifier,
        loss_function=nn.CrossEntropyLoss(),
        test_function=test_bounding_box_classifier,
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

    model = models[args.model].model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_function = models[args.model].loss_function
    test_function = models[args.model].test_function

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
        metrics, test_outputs = test_function(model, test_loader)
        print(metrics)
        log.append(loss_string + "\n")
        log.append(str(metrics) + "\n")

    save_model(args.out_dir, args.model, model, log, train_outputs, test_outputs)
