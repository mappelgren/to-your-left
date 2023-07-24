import argparse
from audioop import avg
from dataclasses import dataclass
from typing import Callable

import torch
from mlt.feature_extractors import DummyFeatureExtractor, ResnetFeatureExtractor
from mlt.image_loader import ClevrImageLoader, FeatureImageLoader
from mlt.preexperiments.data_readers import (
    BasicImageMasker,
    BoundingBoxClassifierDataset,
    CaptionGeneratorDataset,
    Color,
    CoordinatePredictorDataset,
    DaleCaptionAttributeEncoder,
    OneHotAttributeEncoder,
    PreprocessScratch,
    Shape,
    Size,
)
from mlt.preexperiments.models import (
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
from mlt.preexperiments.save import (
    BoundingBoxOutputProcessor,
    CaptionOutputProcessor,
    ModelSaver,
    PixelOutputProcessor,
    StandardOutputProcessor,
)
from mlt.preexperiments.test import (
    BoundingBoxClassifierTester,
    CaptionGeneratorTester,
    CoordinatePredictorTester,
    Tester,
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
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=AttributeCoordinatePredictor,
        model_args={
            "number_colors": len(Color.names()),
            "number_shapes": len(Shape.names()),
            "number_sizes": len(Size.names()),
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
        dataset_args={
            "attribute_encoder": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.APPEND,
                reversed_caption=False,
            )
        },
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=DaleAttributeCoordinatePredictor,
        model_args={
            "vocab_size": len(DaleCaptionAttributeEncoder.vocab),
            "embedding_dim": len(DaleCaptionAttributeEncoder.vocab),
            "encoder_out_dim": len(DaleCaptionAttributeEncoder.vocab),
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
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=AttributeLocationCoordinatePredictor,
        model_args={
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
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
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
        model_args={
            "feature_extractor": ResnetFeatureExtractor(
                number_blocks=4, avgpool=False, fc=False
            )
        },
        loss_function=nn.CrossEntropyLoss(),
        tester=BoundingBoxClassifierTester,
        output_processor=BoundingBoxOutputProcessor,
        output_processor_args={
            "output_fields": ("image_id", "bounding_box", "target_bounding_box")
        },
    ),
    "caption_generator": ModelDefinition(
        dataset=CaptionGeneratorDataset,
        dataset_args={
            "captioner": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.PREPEND,
                reversed_caption=False,
            )
        },
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=CaptionGenerator,
        model_args={
            "image_encoder": ImageEncoder(
                encoder_out_dim=1024,
                feature_extractor=DummyFeatureExtractor(),
            ),
            "caption_decoder": CaptionDecoder(
                vocab_size=len(DaleCaptionAttributeEncoder.vocab),
                embedding_dim=int(len(DaleCaptionAttributeEncoder.vocab) / 2),
                decoder_out_dim=1024,
            ),
            "encoded_sos": DaleCaptionAttributeEncoder.get_encoded_word(
                DaleCaptionAttributeEncoder.SOS_TOKEN
            ),
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
        dataset_args={
            "captioner": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.PREPEND,
                reversed_caption=False,
            ),
            "image_masker": BasicImageMasker(),
        },
        preprocess=ResNet101_Weights.IMAGENET1K_V2.transforms(),
        model=MaskedCaptionGenerator,
        model_args={
            "image_encoder": ImageEncoder(
                encoder_out_dim=1024,
                feature_extractor=DummyFeatureExtractor(),
            ),
            "masked_image_encoder": ImageEncoder(
                encoder_out_dim=1024,
                feature_extractor=DummyFeatureExtractor(),
            ),
            "caption_decoder": CaptionDecoder(
                vocab_size=len(DaleCaptionAttributeEncoder.vocab),
                embedding_dim=len(DaleCaptionAttributeEncoder.vocab),
                decoder_out_dim=2048,
            ),
            "encoded_sos": DaleCaptionAttributeEncoder.get_encoded_word(
                DaleCaptionAttributeEncoder.SOS_TOKEN
            ),
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
    parser.add_argument("--scene_json_dir", type=str, help="Path to the scene json dir")
    parser.add_argument("--image_dir", type=str, help="Path to the scene image dir")
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Path to the hd5 file containing extracted image features",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a saved model state dict",
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

    if args.feature_file is not None:
        image_loader = FeatureImageLoader(
            feature_file=args.feature_file, image_dir=args.image_dir
        )
    else:
        image_loader = ClevrImageLoader(
            image_dir=args.image_dir,
            preprocess=models[args.model].preprocess,
        )

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

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = models[args.model].loss_function

    log = [str(args) + "\n" + str(model) + "\n"]
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
