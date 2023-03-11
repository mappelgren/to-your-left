import argparse
import sys

import egg.core as core
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from architectures import Receiver, Sender
from data_readers import NumberObjectsDataset


def loss(
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
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

def get_params(params):
    parser = argparse.ArgumentParser()

    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--scene_json_file", type=str, default=None, help="Path to the scene JSON file"
    )
    parser.add_argument(
        "--image_folder", type=str, default=None, help="Path to image folder"
    )
    parser.add_argument(
        "--max_number_samples", type=int, default=-1, help="Maximum number of samples (set -1 for all samples)"
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
    dataset = NumberObjectsDataset(scenes_json_file=opts.scene_json_file,
                                   image_path=opts.image_folder,
                                   max_number_samples=opts.max_number_samples)
    n_labels = dataset.get_number_labels()

    train_dataset, test_dataset = random_split(dataset, (80, 20))
    train_loader = DataLoader(
            train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=1
        )
    test_loader = DataLoader(
            test_dataset, batch_size=opts.validation_batch_size, shuffle=True, num_workers=1
        )
    
    receiver = Receiver(opts.receiver_hidden, n_labels)
    sender = Sender(opts.sender_hidden)

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
    
    game = core.SenderReceiverRnnGS(gs_sender, gs_receiver, loss)
    callbacks = [core.TemperatureUpdater(agent=gs_sender, decay=0.9, minimum=0.1)]
    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                # core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    trainer.train(n_epochs=opts.n_epochs)

if __name__ == '__main__':
    main(sys.argv[1:])
