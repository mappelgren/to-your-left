import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from time import gmtime, strftime
from typing import Callable

from egg import core
from mlt.feature_extractors import DummyFeatureExtractor, ResnetFeatureExtractor
from mlt.image_loader import FeatureImageLoader, ImageLoader
from mlt.language_games.callbacks import (
    ExcludingInteractionSaver,
    LogSaver,
    PrintMessages,
    ProgressLogger,
)
from mlt.language_games.data_readers import (
    AttentionPredictorGameBatchIterator,
    BoundingBoxAttentionPredictorGameBatchIterator,
    BoundingBoxCaptionGeneratorGameBatchIterator,
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
    OneHotGeneratorGameBatchIterator,
    OneHotGeneratorGameDataset,
)
from mlt.language_games.models import (
    AttentionPredictorReceiver,
    AttributeSender,
    CaptionGeneratorReceiver,
    CaptionGeneratorSender,
    CoordinatePredictorReceiver,
    DaleAttributeCoordinatePredictorSender,
    DaleAttributeSender,
    DummySender,
    MaskedCoordinatePredictorSender,
    OneHotGeneratorReceiver,
    ReferentialGameReceiver,
    ReferentialGameSender,
)
from mlt.language_games.test import (
    attention_loss,
    captioning_loss,
    classification_loss,
    one_hot_loss,
    pixel_loss,
)
from mlt.preexperiments.data_readers import (
    DaleCaptionAttributeEncoder,
    OneHotAttributeEncoder,
    SingleObjectImageMasker,
)
from mlt.preexperiments.models import CaptionDecoder
from mlt.shared_models import ClevrImageEncoder, CoordinateClassifier
from mlt.util import Persistor, colors, get_model_params, set_model_params
from torch.nn import Module
from torch.utils.data import Dataset, random_split


@dataclass
class ModelDefinition:
    dataset: Dataset
    dataset_args: dict
    split_dataset: bool
    iterator: GameBatchIterator
    image_loader: ImageLoader
    bounding_box_loader: ImageLoader
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
        bounding_box_loader=None,
        iterator=LazaridouReferentialGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=ReferentialGameReceiver,
        receiver_args={},
        loss_function=classification_loss,
    ),
    "discriminator": ModelDefinition(
        dataset=DaleReferentialGameDataset,
        dataset_args={},
        split_dataset=False,
        image_loader=None,
        bounding_box_loader=FeatureImageLoader,
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
        bounding_box_loader=None,
        iterator=CaptionGeneratorGameBatchIterator,
        sender=CaptionGeneratorSender,
        sender_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "masked_image_encoder": ClevrImageEncoder(
                feature_extractor=ResnetFeatureExtractor(
                    pretrained=True,
                    avgpool=False,
                    fc=False,
                    fine_tune=False,
                    number_blocks=3,
                ),
                max_pool=True,
            ),
        },
        receiver=CaptionGeneratorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "caption_decoder": CaptionDecoder,
            "encoded_sos": DaleCaptionAttributeEncoder.get_encoded_word(
                DaleCaptionAttributeEncoder.SOS_TOKEN
            ),
        },
        loss_function=captioning_loss,
    ),
    "bounding_box_caption_generator": ModelDefinition(
        dataset=CaptionGeneratorGameDataset,
        dataset_args={
            "captioner": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.PREPEND,
                reversed_caption=False,
            ),
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=FeatureImageLoader,
        iterator=BoundingBoxCaptionGeneratorGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=CaptionGeneratorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "caption_decoder": CaptionDecoder,
            "encoded_sos": DaleCaptionAttributeEncoder.get_encoded_word(
                DaleCaptionAttributeEncoder.SOS_TOKEN
            ),
        },
        loss_function=captioning_loss,
    ),
    "bounding_box_one_hot_generator": ModelDefinition(
        dataset=OneHotGeneratorGameDataset,
        dataset_args={
            "target_attribute_encoder": OneHotAttributeEncoder(),
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=FeatureImageLoader,
        iterator=OneHotGeneratorGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=OneHotGeneratorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "number_attributes": 13,
        },
        loss_function=one_hot_loss,
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
        bounding_box_loader=None,
        iterator=CoordinatePredictorGameBatchIterator,
        sender=DaleAttributeCoordinatePredictorSender,
        sender_args={
            "sender_encoder_vocab_size": len(DaleCaptionAttributeEncoder.vocab),
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
        },
        receiver=CoordinatePredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
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
        bounding_box_loader=None,
        iterator=CoordinatePredictorGameBatchIterator,
        sender=MaskedCoordinatePredictorSender,
        sender_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "masked_image_encoder": ClevrImageEncoder(
                feature_extractor=ResnetFeatureExtractor(
                    pretrained=True,
                    avgpool=False,
                    fc=False,
                    fine_tune=False,
                    number_blocks=3,
                ),
                max_pool=True,
            ),
        },
        receiver=CoordinatePredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "coordinate_classifier": CoordinateClassifier,
        },
        loss_function=pixel_loss,
    ),
    "masked_attention_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "image_masker": SingleObjectImageMasker(),
            "number_regions": 14,
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=None,
        iterator=AttentionPredictorGameBatchIterator,
        sender=MaskedCoordinatePredictorSender,
        sender_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=True
            ),
            "masked_image_encoder": ClevrImageEncoder(
                feature_extractor=ResnetFeatureExtractor(
                    pretrained=True,
                    avgpool=False,
                    fc=False,
                    fine_tune=False,
                    number_blocks=3,
                ),
                max_pool=True,
            ),
        },
        receiver=AttentionPredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=False
            ),
        },
        loss_function=attention_loss,
    ),
    "dale_attention_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "attribute_encoder": DaleCaptionAttributeEncoder(
                padding_position=DaleCaptionAttributeEncoder.PaddingPosition.APPEND,
                reversed_caption=False,
            ),
            "number_regions": 14,
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=None,
        iterator=AttentionPredictorGameBatchIterator,
        sender=DaleAttributeSender,
        sender_args={
            "sender_encoder_vocab_size": len(DaleCaptionAttributeEncoder.vocab),
            "sender_encoder_embedding": len(DaleCaptionAttributeEncoder.vocab),
            "sender_encoder_out": len(DaleCaptionAttributeEncoder.vocab),
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=False
            ),
        },
        receiver=AttentionPredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=False
            ),
        },
        loss_function=attention_loss,
    ),
    "attribute_attention_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "attribute_encoder": OneHotAttributeEncoder(),
            "number_regions": 14,
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=None,
        iterator=AttentionPredictorGameBatchIterator,
        sender=AttributeSender,
        sender_args={},
        receiver=AttentionPredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=False
            ),
        },
        loss_function=attention_loss,
    ),
    "bounding_box_attention_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "number_regions": 14,
        },
        split_dataset=False,
        image_loader=FeatureImageLoader,
        bounding_box_loader=FeatureImageLoader,
        iterator=BoundingBoxAttentionPredictorGameBatchIterator,
        sender=ReferentialGameSender,
        sender_args={},
        receiver=AttentionPredictorReceiver,
        receiver_args={
            "image_encoder": ClevrImageEncoder(
                feature_extractor=DummyFeatureExtractor(), max_pool=False
            ),
        },
        loss_function=attention_loss,
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
        "--image_feature_file",
        type=str,
        default=None,
        help="Path to the hd5 file containing extracted image features",
    )
    parser.add_argument(
        "--bounding_box_feature_file",
        type=str,
        default=None,
        help="Path to the hd5 file containing extracted bounding box features",
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
    parser.add_argument(
        "--baseline",
        default=False,
        action="store_true",
        help="sender sends random messages",
    )

    # -- TRAINING --
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )

    # -- AGENTS --
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_image_embedding",
        type=int,
        help="Output dimensionality of the layer that embeds the image in Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_encoder_out",
        type=int,
        help="Size of the LSTM encoder of Sender when attributes are encoded with descriptions (default: 10)",
    )
    parser.add_argument(
        "--sender_coordinate_classifier",
        type=int,
        help="Dimensions for the coordinate predictor for Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_projection",
        type=int,
        help="Projection dimension to combin message and image (default: 10)",
    )

    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_image_embedding",
        type=int,
        help="Output dimensionality of the layer that embeds the image for Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_decoder_embedding",
        type=int,
        help="Output dimensionality of the captioning tokens (default: 10)",
    )
    parser.add_argument(
        "--receiver_decoder_out",
        type=int,
        help="Output dimensionality of the layer that embeds the caption for Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_projection",
        type=int,
        help="Projection dimension to combin message and image (default: 10)",
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
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    model = models[opts.model]

    image_dir = os.path.join(opts.dataset_base_dir, datasets[opts.dataset], "images/")
    scene_json_dir = os.path.join(
        opts.dataset_base_dir, datasets[opts.dataset], "scenes/"
    )

    if model.image_loader:
        image_feature_file = os.path.join(
            opts.dataset_base_dir,
            datasets[opts.dataset],
            "features",
            opts.image_feature_file,
        )
        image_loader = model.image_loader(
            feature_file=image_feature_file, image_dir=image_dir
        )
    else:
        image_loader = None

    if model.bounding_box_loader:
        bounding_box_feature_file = os.path.join(
            opts.dataset_base_dir,
            datasets[opts.dataset],
            "features",
            opts.bounding_box_feature_file,
        )
        bounding_box_loader = model.bounding_box_loader(
            feature_file=bounding_box_feature_file, image_dir=image_dir
        )
    else:
        bounding_box_loader = None

    dataset_args = {
        "scenes_json_dir": scene_json_dir,
        "image_loader": image_loader,
        "bounding_box_loader": bounding_box_loader,
        "max_number_samples": opts.max_samples,
        "data_root_dir": opts.data_root_dir,
        **model.dataset_args,
    }

    dataset_identifier = hashlib.sha256(
        str(f"{model.dataset.__name__}({dataset_args})").encode()
    ).hexdigest()
    dataset_dir = os.path.join(opts.out_dir, "datasets")
    dataset_file = os.path.join(dataset_dir, f"{dataset_identifier}.h5")
    persistor = Persistor(dataset_file)
    if os.path.exists(dataset_file):
        print(f"Loading dataset {dataset_identifier}...", end="\r")
        dataset = persistor.load(model.dataset)
        print(f"Dataset {dataset_identifier} loaded.   ")
    else:
        dataset = model.dataset.load(**dataset_args, persistor=persistor)
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

    if opts.baseline:
        sender = DummySender
    else:
        sender = model.sender

    sender_params = get_model_params(sender, opts, "sender_")
    sender_args = set_model_params(model.sender_args, sender_params)

    receiver_params = get_model_params(model.receiver, opts, "receiver_")
    receiver_args = set_model_params(model.receiver_args, receiver_params)

    appendices = []
    params_check = True
    message_params = {
        "vocab_size": opts.vocab_size,
        "max_len": opts.max_len,
        "sender_hidden": opts.sender_hidden,
        "sender_embedding": opts.sender_embedding,
        "receiver_hidden": opts.receiver_hidden,
        "receiver_embedding": opts.receiver_embedding,
    }
    for param, value in sorted((sender_args | receiver_args | message_params).items()):
        if param not in opts:
            continue

        if value is None:
            color = colors.RED
            params_check = False
        else:
            color = colors.GREEN

        appendices.append((param, value))

        print(f"{param} = {color}{value}{colors.ENDC}")

    if not params_check:
        return

    if "caption_decoder" in receiver_args.keys():
        receiver_args["caption_decoder"] = receiver_args["caption_decoder"](
            decoder_embedding=receiver_args["receiver_decoder_embedding"],
            decoder_out=receiver_args["receiver_decoder_out"],
            decoder_vocab_size=len(DaleCaptionAttributeEncoder.vocab),
        )
    if "coordinate_classifier" in receiver_args.keys():
        receiver_args["coordinate_classifier"] = receiver_args["coordinate_classifier"](
            classifier_dimension=receiver_args["coordinate_classifier"]
        )

    receiver = model.receiver(**receiver_args)
    sender = sender(**sender_args)

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
            callbacks.extend(
                [
                    core.ConsoleLogger(
                        print_train_loss=True,
                    ),
                    ProgressLogger(batches_per_epoch=opts.batches_per_epoch),
                ]
            )

    if opts.save:
        time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        baseline = "_baseline" if opts.baseline else ""
        save_appendix = "_".join([str(value) for _, value in appendices])
        out_dir = os.path.join(
            opts.out_dir,
            f"{time}_{opts.model}{baseline}_{opts.dataset}_{save_appendix}",
        )
        callbacks.extend(
            [
                # core.CheckpointSaver(
                #     checkpoint_path=out_dir,
                #     prefix="checkpoint",
                #     checkpoint_freq=0,
                # ),
                ExcludingInteractionSaver(
                    checkpoint_dir=out_dir,
                    train_epochs=[opts.n_epochs],
                    test_epochs=[opts.n_epochs],
                ),
                LogSaver(
                    out_dir=out_dir,
                    command=str(opts),
                    variables=[param for param, _ in appendices],
                    sender=sender,
                    receiver=receiver,
                ),
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
