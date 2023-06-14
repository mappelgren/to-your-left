import argparse
import sys

import egg.core as core
import torch
import torch.nn.functional as F
from mlt.language_games.data_readers import (
    DaleReferentialGameDataset,
    LazaridouReferentialGameDataset,
    LazaridouReferentialGameLoader,
)
from mlt.language_games.models import ReferentialGameReceiver, ReferentialGameSender
from mlt.preexperiments.feature_extractors import ResnetFeatureExtractor
from torch.utils.data import DataLoader, random_split


def classification_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.nll_loss(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def get_params(params):
    parser = argparse.ArgumentParser()

    # -- DATASET --
    parser.add_argument(
        "--data_root_path", type=str, default=None, help="Path to the scene json dir"
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
    # arguments concerning the training method
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
    # arguments concerning the agent architectures
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
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    dataset = DaleReferentialGameDataset(
        data_root_path=opts.data_root_path,
        feature_extractor=ResnetFeatureExtractor(),
        max_number_samples=opts.max_samples,
        device=opts.device,
    )

    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(
        dataset, (train_dataset_length, test_dataset_length)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opts.validation_batch_size,
        shuffle=True,
        num_workers=1,
    )

    # dataset = LazaridouReferentialGameDataset(
    #     data_root_path=opts.data_root_path,
    # )

    # train_loader = LazaridouReferentialGameLoader(
    #     dataset=dataset,
    #     batch_size=opts.batch_size,
    #     batches_per_epoch=opts.batches_per_epoch,
    #     seed=None,
    # )
    # test_loader = LazaridouReferentialGameLoader(
    #     dataset=dataset,
    #     batch_size=opts.validation_batch_size,
    #     batches_per_epoch=opts.batches_per_epoch,
    #     seed=7,
    # )

    receiver = ReferentialGameReceiver(opts.receiver_embedding)
    sender = ReferentialGameSender(opts.sender_hidden, opts.sender_embedding)

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

    game = core.SenderReceiverRnnGS(gs_sender, gs_receiver, classification_loss)

    callbacks = [core.TemperatureUpdater(agent=gs_sender, decay=0.9, minimum=0.1)]
    if opts.print_validation_events:
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))

    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks
        + [
            core.ConsoleLogger(print_train_loss=True, as_json=True),
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    main(sys.argv[1:])
