import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from time import gmtime, strftime
from typing import Callable

import torch
from egg import core
from mlt.feature_extractors import DummyFeatureExtractor, ResnetFeatureExtractor
from mlt.image_loader import FeatureImageLoader, ImageLoader
from mlt.language_games.callbacks import (
    ExcludingInteractionSaver,
    LogSaver,
    PrintMessages,
)
from mlt.language_games.data_readers import (
    CaptionGeneratorGameBatchIterator,
    CaptionGeneratorGameDataset,
    CoordinatePredictorGameBatchIterator,
    CoordinatePredictorGameDataset,
    DaleReferentialGameBatchIterator,
    DaleReferentialGameDataset,
    GameBatchIterator,
    GameLoader,
    LazaridouReferentialGameBatchIterator,
    LazaridouReferentialGameDataset,
)
from mlt.language_games.models import (
    CaptionGeneratorReceiver,
    CaptionGeneratorSender,
    CoordinatePredictorReceiver,
    DaleAttributeCoordinatePredictorSender,
    DummyReferentialSender,
    MaskedCoordinatePredictorSender,
    ReferentialGameReceiver,
    ReferentialGameSender,
)
from mlt.language_games.test import captioning_loss, classification_loss, pixel_loss
from mlt.preexperiments.data_readers import (
    DaleCaptionAttributeEncoder,
    SingleObjectImageMasker,
)
from mlt.preexperiments.models import CaptionDecoder
from mlt.shared_models import ClevrImageEncoder, CoordinateClassifier
from torch.nn import Module
from torch.utils.data import Dataset, random_split


@dataclass
class ModelDefinition:
    dataset: Dataset
    dataset_args: dict
    split_dataset: bool
    iterator: GameBatchIterator
    image_loader: ImageLoader
    sender: Module
    sender_args: dict
    receiver: Module
    receiver_args: dict
    loss_function: Callable


models = {
    "lazaridou": ModelDefinition(
        dataset=LazaridouReferentialGameDataset,
        dataset_args={},
        split_dataset=False,
        image_loader=None,
        iterator=LazaridouReferentialGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=ReferentialGameReceiver,
        receiver_args={},
        loss_function=classification_loss,
    ),
    "test_discriminator": ModelDefinition(
        dataset=DaleReferentialGameDataset,
        dataset_args={},
        split_dataset=False,
        image_loader=FeatureImageLoader,
        iterator=DaleReferentialGameBatchIterator,
        sender=DummyReferentialSender,
        sender_args={},
        receiver=ReferentialGameReceiver,
        receiver_args={},
        loss_function=classification_loss,
    ),
    "discriminator": ModelDefinition(
        dataset=DaleReferentialGameDataset,
        dataset_args={},
        split_dataset=False,
        image_loader=FeatureImageLoader,
        iterator=DaleReferentialGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=ReferentialGameReceiver,
        receiver_args={},
        loss_function=classification_loss,
    ),
    "caption_generator": ModelDefinition(
        dataset=CaptionGeneratorGameDataset,
        dataset_args={
            "captioner": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.PREPEND,
                reversed_caption=False,
            ),
            "image_masker": SingleObjectImageMasker(),
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        iterator=CaptionGeneratorGameBatchIterator,
        sender=CaptionGeneratorSender,
        sender_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "masked_image_encoder": ClevrImageEncoder(
                feature_extractor=ResnetFeatureExtractor(
                    pretrained=True,
                    avgpool=False,
                    fc=False,
                    fine_tune=False,
                    number_blocks=3,
                ),
            ),
            "embedding_dimension": 2048,
        },
        receiver=CaptionGeneratorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "embedding_dimension": 1024,
            "caption_decoder": CaptionDecoder,
            "encoded_sos": DaleCaptionAttributeEncoder.get_encoded_word(
                DaleCaptionAttributeEncoder.SOS_TOKEN
            ),
        },
        loss_function=captioning_loss,
    ),
    "dale_attribute_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "attribute_encoder": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.APPEND,
                reversed_caption=False,
            )
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        iterator=CoordinatePredictorGameBatchIterator,
        sender=DaleAttributeCoordinatePredictorSender,
        sender_args={
            "vocab_size": len(DaleCaptionAttributeEncoder.vocab),
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "image_embedding_dimension": 1024,
        },
        receiver=CoordinatePredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "embedding_dimension": 1024,
            "coordinate_classifier": CoordinateClassifier,
        },
        loss_function=pixel_loss,
    ),
    "masked_coordinate_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "image_masker": SingleObjectImageMasker(),
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        iterator=CoordinatePredictorGameBatchIterator,
        sender=MaskedCoordinatePredictorSender,
        sender_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "masked_image_encoder": ClevrImageEncoder(
                feature_extractor=ResnetFeatureExtractor(
                    pretrained=True,
                    avgpool=False,
                    fc=False,
                    fine_tune=False,
                    number_blocks=3,
                ),
            ),
            "embedding_dimension": 2048,
        },
        receiver=CoordinatePredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(),
            ),
            "embedding_dimension": 1024,
            "coordinate_classifier": CoordinateClassifier,
        },
        loss_function=pixel_loss,
    ),
}

# names of the datasets and their foldernames
datasets = {
    "dale-2": "clevr-images-unambigous-dale-two",
    "dale-5": "clevr-images-unambigous-dale",
    "single": "clevr-images-random-single",
    "colour": "clevr-images-unambigous-colour",
}


def get_params(params):
    parser = argparse.ArgumentParser()

    # -- DATASET --
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        help="Path to the base directory of all datasets",
    )
    parser.add_argument("--dataset", choices=datasets.keys(), help="datasets, to load")
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Path to the hd5 file containing extracted image features",
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default=None,
        help="Path to root dir of the specific dataset",
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="max samples to load"
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="batches shown to the model every epoch",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    parser.add_argument(
        "--validation_batches_per_epoch",
        type=int,
        default=100,
        help="batches shown to the model every epoch",
    )

    # -- MODEL --
    parser.add_argument(
        "--model",
        choices=models.keys(),
        help="model to load",
    )

    # -- TRAINING --
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )

    # -- AGENTS --
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_image_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the image in Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_encoder_dim",
        type=int,
        default=10,
        help="Size of the LSTM encoder of Sender when attributes are encoded with descriptions (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_decoder_out_dim",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the caption for Receiver (default: 10)",
    )
    parser.add_argument(
        "--coordinate_classifier_dimension",
        type=int,
        default=10,
        help="Dimensions for the coordinate predictor for Receiver (default: 10)",
    )

    # -- OUTPUT --
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save models to checkpoint",
    )
    parser.add_argument(
        "--progress_bar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="display progress bar",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/",
        help="directory, where the output should be saved",
    )
    parser.add_argument(
        "--save_appendix",
        type=str,
        default="",
        help="information that will be appended to the name of the folder",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    image_dir = os.path.join(opts.dataset_base_dir, datasets[opts.dataset], "images/")
    scene_json_dir = os.path.join(
        opts.dataset_base_dir, datasets[opts.dataset], "scenes/"
    )
    feature_file = os.path.join(
        opts.dataset_base_dir, datasets[opts.dataset], "features", opts.feature_file
    )

    model = models[opts.model]

    if model.image_loader:
        image_loader = model.image_loader(
            feature_file=feature_file, image_dir=image_dir
        )
    else:
        image_loader = None

    dataset_args = {
        "scenes_json_dir": scene_json_dir,
        "image_loader": image_loader,
        "max_number_samples": opts.max_samples,
        "data_root_dir": opts.data_root_dir,
        **model.dataset_args,
    }

    dataset_identifier = hashlib.sha256(
        str(f"{model.dataset.__name__}({dataset_args})").encode()
    ).hexdigest()
    dataset_dir = os.path.join(opts.out_dir, "datasets")
    dataset_file = os.path.join(dataset_dir, f"{dataset_identifier}.h5")
    if os.path.exists(dataset_file):
        print(f"Loading dataset {dataset_identifier}...", end="\r")
        dataset = model.dataset.load_file(dataset_file)
        print(f"Dataset {dataset_identifier} loaded.   ")
    else:
        dataset = model.dataset.load(**dataset_args)

        print(f"Saving dataset {dataset_identifier}...", end="\r")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset.save(dataset_file)
        print(f"Dataset {dataset_identifier} saved.   ")

    if model.split_dataset:
        train_dataset_length = int(0.8 * len(dataset))
        test_dataset_length = len(dataset) - train_dataset_length
        train_dataset, test_dataset = random_split(
            dataset, (train_dataset_length, test_dataset_length)
        )
    else:
        train_dataset = test_dataset = dataset

    train_loader = GameLoader(
        dataset=train_dataset,
        iterator=model.iterator,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        train_mode=True,
        seed=None,
    )
    test_loader = GameLoader(
        dataset=test_dataset,
        iterator=model.iterator,
        batch_size=opts.validation_batch_size,
        batches_per_epoch=opts.validation_batches_per_epoch,
        train_mode=False,
        seed=7,
    )

    sender_args = model.sender_args
    sender_args["embedding_dimension"] = opts.sender_embedding
    sender_args["image_embedding_dimension"] = opts.sender_image_embedding
    sender_args["hidden_size"] = opts.sender_hidden
    sender_args["encoder_out_dim"] = opts.sender_encoder_dim

    receiver_args = model.receiver_args
    receiver_args["embedding_dimension"] = opts.receiver_embedding

    if "caption_decoder" in receiver_args.keys():
        receiver_args["caption_decoder"] = receiver_args["caption_decoder"](
            embedding_dim=opts.receiver_embedding,
            decoder_out_dim=opts.receiver_decoder_out_dim,
            vocab_size=len(DaleCaptionAttributeEncoder.vocab),
        )
    if "coordinate_classifier" in receiver_args.keys():
        receiver_args["coordinate_classifier"] = receiver_args["coordinate_classifier"](
            classifier_dimension=opts.coordinate_classifier_dimension
        )

    receiver = model.receiver(**receiver_args)
    sender = model.sender(**sender_args)

    gs_sender = core.RnnSenderGS(
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        temperature=opts.temperature,
    )

    gs_receiver = core.RnnReceiverGS(
        receiver,
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell,
    )

    game = core.SenderReceiverRnnGS(gs_sender, gs_receiver, model.loss_function)

    callbacks = [core.TemperatureUpdater(agent=gs_sender, decay=0.9, minimum=0.1)]
    if opts.print_validation_events:
        callbacks.extend(
            [
                PrintMessages(n_epochs=opts.n_epochs),
                core.ConsoleLogger(
                    print_train_loss=True,
                ),
            ]
        )
    else:
        if opts.progress_bar:
            callbacks.append(
                core.ProgressBarLogger(
                    n_epochs=opts.n_epochs,
                    train_data_len=opts.batches_per_epoch,
                    test_data_len=opts.batches_per_epoch,
                )
            )
        else:
            callbacks.append(
                core.ConsoleLogger(
                    print_train_loss=True,
                ),
            )

    if opts.save:
        out_dir = os.path.join(
            opts.out_dir,
            f"{strftime('%Y-%m-%d_%H-%M-%S', gmtime())}_{opts.model}_{opts.dataset}{'_' + opts.save_appendix if opts.save_appendix != '' else ''}",
        )
        callbacks.extend(
            [
                core.CheckpointSaver(
                    checkpoint_path=out_dir,
                    prefix="checkpoint",
                    checkpoint_freq=0,
                ),
                ExcludingInteractionSaver(
                    checkpoint_dir=out_dir,
                    train_epochs=[opts.n_epochs],
                    test_epochs=[opts.n_epochs],
                ),
                LogSaver(out_dir=out_dir, command=str(opts)),
            ]
        )

    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    main(sys.argv[1:])
