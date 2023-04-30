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
    BasicImageMasker,
    BoundingBoxClassifierDataset,
    CaptionGeneratorDataset,
    CoordinatePredictorDataset,
    DaleAttributeEncoder,
    OneHotAttributeEncoder,
    PreprocessScratch,
)
from feature_extractors import (
    DummyFeatureExtractor,
    ResnetFeatureExtractor,
    VggFeatureExtractor,
)
from image_loader import ClevrImageLoader, FeatureImageLoader
from models import (
    AttributeCoordinatePredictor,
    AttributeLocationCoordinatePredictor,
    BoundingBoxClassifier,
    CaptionDecoder,
    CaptionGenerator,
    CoordinatePredictor,
    DaleAttributeCoordinatePredictor,
    ImageEncoder,
    MaskedCaptionGenerator,
    MaskedCoordinatePredictor,
)
from save import (
    BoundingBoxOutputProcessor,
    CaptionOutputProcessor,
    ModelSaver,
    PixelOutputProcessor,
    StandardOutputProcessor,
)
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torcheval.metrics import Mean
from torchvision.models import ResNet101_Weights


def pixel_loss(model_output, ground_truth):
    loss = torch.diagonal(torch.cdist(model_output, ground_truth.float()))

    return torch.mean(loss)


@dataclass
class ModelDefinition:
    dataset: Dataset
    dataset_args: dict
    preprocess: Callable
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
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=CoordinatePredictor,
        model_args={"feature_extractor": DummyFeatureExtractor()},
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "coordinate_predictor_scratch": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={},
        preprocess=PreprocessScratch(250),
        model=CoordinatePredictor,
        model_args={
            "feature_extractor": ResnetFeatureExtractor(
                pretrained=False, fine_tune=True
            )
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"attribute_encoder": OneHotAttributeEncoder()},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=AttributeCoordinatePredictor,
        model_args={
            "number_colors": 8,
            "number_shapes": 3,
            "number_sizes": 2,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "dale_attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"attribute_encoder": DaleAttributeEncoder()},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=DaleAttributeCoordinatePredictor,
        model_args={
            "vocab_size": 14,
            "embedding_dim": 32,
            "encoder_out_dim": 32,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "feature_dale_attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={
            "attribute_encoder": DaleAttributeEncoder(),
            "feature_extractor": ResnetFeatureExtractor(
                pretrained=True, fine_tune=False
            ),
        },
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=DaleAttributeCoordinatePredictor,
        model_args={
            "vocab_size": 14,
            "embedding_dim": 32,
            "encoder_out_dim": 32,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "feature_dale_vgg_attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={
            "attribute_encoder": DaleAttributeEncoder(),
            "feature_extractor": VggFeatureExtractor(pretrained=True, fine_tune=False),
        },
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=DaleAttributeCoordinatePredictor,
        model_args={
            "vocab_size": 14,
            "embedding_dim": 32,
            "encoder_out_dim": 32,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "dale_vgg_attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"attribute_encoder": DaleAttributeEncoder()},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=DaleAttributeCoordinatePredictor,
        model_args={
            "vocab_size": 14,
            "embedding_dim": 32,
            "encoder_out_dim": 32,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "attribute_location_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={
            "attribute_encoder": OneHotAttributeEncoder(),
            "encode_locations": True,
        },
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=AttributeLocationCoordinatePredictor,
        model_args={
            "number_colors": 8,
            "number_shapes": 3,
            "number_sizes": 2,
            "feature_extractor": DummyFeatureExtractor(),
        },
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "masked_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorDataset,
        dataset_args={"image_masker": BasicImageMasker()},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=MaskedCoordinatePredictor,
        model_args={"feature_extractor": DummyFeatureExtractor()},
        loss_function=pixel_loss,
        tester=CoordinatePredictorTester,
        output_processor=PixelOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "x", "y", "target_x", "target_y")
        },
    ),
    "bounding_box_classifier": ModelDefinition(
        dataset=BoundingBoxClassifierDataset,
        dataset_args={},
        preprocess=PreprocessScratch(50),
        model=BoundingBoxClassifier,
        model_args={"feature_extractor": DummyFeatureExtractor()},
        loss_function=nn.CrossEntropyLoss(),
        tester=BoundingBoxClassifierTester,
        output_processor=BoundingBoxOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "bounding_box", "target_bounding_box")
        },
    ),
    "caption_generator": ModelDefinition(
        dataset=CaptionGeneratorDataset,
        dataset_args={},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=CaptionGenerator,
        model_args={
            "image_encoder": ImageEncoder(
                2048,
                feature_extractor=DummyFeatureExtractor(),
            ),
            "caption_decoder": CaptionDecoder(14, 128, 2048),
            "encoded_sos": 0,
        },
        loss_function=nn.CrossEntropyLoss(),
        tester=CaptionGeneratorTester,
        output_processor=CaptionOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "caption", "target_caption")
        },
    ),
    "masked_caption_generator": ModelDefinition(
        dataset=CaptionGeneratorDataset,
        dataset_args={"image_masked": BasicImageMasker()},
        preprocess=ResNet101_Weights.DEFAULT.transforms(),
        model=MaskedCaptionGenerator,
        model_args={
            "image_encoder": ImageEncoder(
                2048,
                feature_extractor=DummyFeatureExtractor(),
            ),
            "caption_decoder": CaptionDecoder(14, 128, 4096),
            "encoded_sos": 0,
        },
        loss_function=nn.CrossEntropyLoss(),
        tester=CaptionGeneratorTester,
        output_processor=CaptionOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "caption", "target_caption")
        },
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
        "--feature_file",
        type=str,
        default=None,
        help="Path to the hd5 file containing extracted image features",
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
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
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

    if args.image_dir is not None:
        image_loader = ClevrImageLoader(
            image_dir=args.image_dir,
            preprocess=models[args.model].preprocess,
        )
    elif args.feature_file is not None:
        image_loader = FeatureImageLoader(args.feature_file)
    else:
        raise AttributeError("either image path or feature file must be set")

    dataset = models[args.model].dataset(
        scenes_json_dir=args.scene_json_dir,
        image_loader=image_loader,
        max_number_samples=args.max_samples,
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = models[args.model].loss_function

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
            train_outputs.extend(zip(image_id, output.detach(), ground_truth))

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
