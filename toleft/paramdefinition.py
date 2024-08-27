
import argparse


from dataclasses import dataclass


from egg import core

from feature_extractors import DummyFeatureExtractor, ResnetFeatureExtractor
from image_loader import ImageLoader, FeatureImageLoader
from data_readers import GameBatchIterator, CoordinatePredictorGameDataset, \
    AttentionPredictorGameBatchIterator
from models import MaskedCoordinatePredictorSender, AttentionPredictorReceiver
from torch.utils.data import Dataset
from torch.nn import Module
from typing import Callable

from data_readers import SingleObjectImageMasker
from test import attention_loss
from shared_models import ClevrImageEncoder

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


models = {"masked_attention_predictor": ModelDefinition(
        dataset=CoordinatePredictorGameDataset,
        dataset_args={
            "image_masker": SingleObjectImageMasker(),
            "number_regions": 14,
        },
        split_dataset=True,
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
    ),}

datasets = {
    "dale-2": "clevr-images-unambigous-dale-two",
    "dale-5": "clevr-images-unambigous-dale",
    "single": "clevr-images-random-single",
    "colour": "clevr-images-unambigous-colour",
    "spatial": "clevr-spatial"
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
    parser.add_argument(
        "--receiver_coordinate_classifier_dimension",
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
