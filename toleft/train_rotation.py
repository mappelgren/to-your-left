# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.nn.functional as F

import egg.core as core
from egg.zoo.signal_game.archs import InformedSender, Receiver
from torch.onnx.symbolic_opset9 import tensor

from egg.core import Callback, InteractionSaver
from rotation_features import RotFeat, RotationLoader




def parse_arguments():
    parser = argparse.ArgumentParser()
    arser.add_argument("--root", default="/home/xappma/spatial-dataset", help="data root folder")
    # 2-agents specific parameters
    parser.add_argument(
        "--tau_s", type=float, default=10.0, help="Sender Gibbs temperature"
    )
    parser.add_argument(
        "--game_size", type=int, default=2, help="Number of images seen by an agent"
    )
    parser.add_argument("--same", type=int, default=0, help="Use same concepts")
    parser.add_argument("--embedding_size", type=int, default=50, help="embedding size")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="hidden size (number of filters informed sender)",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="Batches in a single training/validation epoch",
    )
    parser.add_argument("--inf_rec", type=int, default=0, help="Use informed receiver")
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.",
    )
    parser.add_argument("--gs_tau", type=float, default=1.0, help="GS temperature")

    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}


def get_game(opt):
    feat_size = opt.featsize
    sender = InformedSender(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.hidden_size,
        opt.vocab_size,
        temp=opt.tau_s,
    )
    receiver = Receiver(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.vocab_size,
        reinforce=(opt.mode == "rf"),
    )
    if opt.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.01,
        )
    elif opt.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opt.mode}")

    return game

class Opts(object):
    def __init__(self, root, batch_size=32, batches_per_epoch=30, mode='rl', n_epochs=10, random_seed=None, checkpoint_dir=None,
             preemptable=False, checkpoint_freq=0, validation_freq=1, load_from_checkpoint=None, no_cuda=False,
             optimizer='adam', lr=1e-2, update_freq=1, vocab_size=10, max_len=1, tensorboard=False, tensorboard_dir="runs/",
             distributed_port=18363, fp16=False, tau_s=10, game_size=2, hidden_size=20, embedding_size=50, featsize=1000):

        self.random_seed = random_seed  # set random seed
        self.checkpoint_dir = checkpoint_dir  # str where the checkpoints are stored
        self.preemptable = preemptable  # bool if the flag is set trainer would always try to initiialise itself from a checkpoint
        self.checkpoint_freq = checkpoint_freq  # int how often the checkpoints are saved
        self.validation_freq = validation_freq  # int the validation would be run every validation_freq epochs
        self.n_epochs = n_epochs  # int number of epochs to train
        self.load_from_checkpoint = load_from_checkpoint  # str if the parameter is set then trainer, model, and optimiser states are loaded from the checkpoint
        self.no_cuda = no_cuda  # disable cuda
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.batch_size = batch_size  # input batchsize for training
        self.optimizer = optimizer  # [adam, sgd, adagrad]
        self.lr = lr  # float learning rate
        self.update_freq = update_freq  # weights are updated every update_freq batches
        self.vocab_size = vocab_size  # number of symbols in the vocabulary including end of sequence symbol
        self.max_len = max_len  # max length of sequence
        self.tensorboard = tensorboard  # enable tensorboard
        self.tensorboard_dir = tensorboard_dir  # path for tensorboard log
        self.distrubted_port = distributed_port  # port to use in distributed learning
        self.fp16 = fp16  # use mixed-precision for training/evaluating models
        self.batches_per_epoch= batches_per_epoch
        self.mode = mode
        self.tau_s = tau_s
        self.game_size = game_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')
        self.featsize = featsize

def run_game(root='/home/xappma/to-your-left/data', dataset='one_colour_square', features_name='VGG-pool_no-layers_3-fc_no-bb_no.h5py',
             validation_dataset=None, batch_size=32, batches_per_epoch=30, mode='rf', n_epochs=10, random_seed=None, checkpoint_dir=None,
             preemptable=False, checkpoint_freq=0, validation_freq=1, load_from_checkpoint=None, no_cuda=False,
             optimizer='adam', lr=1e-2, update_freq=1, vocab_size=10, max_len=1, tensorboard=False, tensorboard_dir="runs/",
             distributed_port=18363, fp16=False, tau_s=10, game_size=2, hidden_size=20, embedding_size=50, featsize=1000):

    opts = Opts(root, batch_size, batches_per_epoch, mode, n_epochs, random_seed, checkpoint_dir,
                preemptable, checkpoint_freq, validation_freq, load_from_checkpoint, no_cuda, optimizer, lr, update_freq,
                vocab_size, max_len, tensorboard, tensorboard_dir, distributed_port, fp16,
                tau_s, game_size, hidden_size, embedding_size, featsize)



    opts = core.init(opts)

    train_data = RotFeat(root=root, dataset=dataset, name=features_name)
    train_loader = RotationLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=batches_per_epoch,
        seed=None,
    )

    validation_set = dataset if validation_dataset is None else validation_dataset
    valid_data = RotFeat(root=root, dataset=validation_set, name=features_name)
    validation_loader = RotationLoader(
        valid_data,
        opt=opts,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        seed=7,
    )
    game = get_game(opts)
    optimizer = core.build_optimizer(game.parameters(), opts)

    if mode == "gs":
        callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    callbacks.append(InteractionSaver(train_epochs=[10], test_epochs=[10], checkpoint_dir='checkpoints/'))
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=n_epochs)

    core.close()


if __name__ == "__main__":
    opts = parse_arguments()

    data_folder = os.path.join(opts.root, "features/")
    dataset = RotFeat(root=data_folder)

    train_loader = RotationLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=None,
    )
    validation_loader = RotationLoader(
        dataset,
        opt=opts,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        seed=7,
    )
    game = get_game(opts)
    optimizer = core.build_optimizer(game.parameters())
    callback = None
    if opts.mode == "gs":
        callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []


    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    callbacks.append()
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=opts.n_epochs)

    core.close()
