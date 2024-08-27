import hashlib
import os
import sys
from dataclasses import dataclass
from time import strftime, gmtime

from egg import core

from feature_extractors import DummyFeatureExtractor, ResnetFeatureExtractor
from image_loader import ImageLoader, FeatureImageLoader
from callbacks import PrintMessages, ProgressLogger, ExcludingInteractionSaver, LogSaver
from data_readers import GameBatchIterator, CoordinatePredictorGameDataset, \
    AttentionPredictorGameBatchIterator, GameLoader
from models import MaskedCoordinatePredictorSender, AttentionPredictorReceiver, DummySender
from torch.utils.data import Dataset, random_split
from torch.nn import Module
from typing import Callable

from data_readers import SingleObjectImageMasker, DaleCaptionAttributeEncoder
from test import attention_loss
from shared_models import ClevrImageEncoder
from util import Persistor, get_model_params, set_model_params, colors

from paramdefinition import models, datasets, get_params

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




my_model = ModelDefinition(
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
    )


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    image_dir = os.path.join(opts.dataset_base_dir, datasets[opts.dataset], "images/")
    scene_json_dir = os.path.join(
        opts.dataset_base_dir, datasets[opts.dataset], "scenes/"
    )

    model = models[opts.model]

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

    print("dataset length")
    print(len(dataset))

    if model.split_dataset:
        train_dataset_length = int(0.8 * len(dataset))
        test_dataset_length = len(dataset) - train_dataset_length
        train_dataset, test_dataset = random_split(
            dataset, (train_dataset_length, test_dataset_length)
        )
    else:
        train_dataset = test_dataset = dataset

    print("test dataset")
    print(test_dataset.__len__())

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
    print(test_dataset)
    print(model.iterator)
    print(opts.validation_batch_size)
    print(opts.validation_batches_per_epoch)

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
            coordinate_classifier_dimension=receiver_args[
                "receiver_coordinate_classifier_dimension"
            ]
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

