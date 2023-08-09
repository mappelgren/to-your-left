import json
import os
import pathlib
import pprint

import torch
from egg.core import Callback, InteractionSaver
from egg.core.interaction import Interaction


class LogSaver(Callback):
    def __init__(self, out_dir: str, command: str) -> None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.file_path = os.path.join(out_dir, "log.txt")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(pprint.pformat(command) + "\n")

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self._save_log(loss, logs, "train", epoch)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self._save_log(loss, logs, "test", epoch)

    def _save_log(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        dump.update(dict(mode=mode, epoch=epoch))
        output_message = json.dumps(dump)

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(output_message + "\n")


class ExcludingInteractionSaver(InteractionSaver):
    @staticmethod
    def dump_interactions(
        logs: Interaction,
        mode: str,
        epoch: int,
        rank: int,
        dump_dir: str = "./interactions",
    ):
        dump_dir = pathlib.Path(dump_dir) / mode / f"epoch_{epoch}"
        dump_dir.mkdir(exist_ok=True, parents=True)

        # exclude space intensive information that is retrievable from the dataset
        logs.sender_input = torch.tensor(0)
        logs.receiver_input = torch.tensor(0)

        if logs.aux_input is not None:
            to_remove = ["masked_image", "caption", "train_mode", "attribute_tensor"]
            for key in to_remove:
                if key in logs.aux_input:
                    logs.aux_input[key] = torch.tensor(0)

        torch.save(logs, dump_dir / f"interaction_gpu{rank}")


# The PrintValidationEvents callback function checks that we are at the
# last epoch (either the last epoch required by the user, or because
# of early stopping), and it prints sender_input, labels, message and
# receiver_output for all data points in the validation set.
# These data are stored in an Interaction object (see interaction.py
# under core), that logs various data about each game data point.
class PrintMessages(Callback):
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = n_epochs

    @staticmethod
    def print_events(logs: Interaction):
        print("MESSAGES")
        print([m.tolist() for m in logs.message], sep="\n")

    # here is where we make sure we are printing the validation set (on_validation_end, not on_epoch_end)
    def on_validation_end(self, _loss, logs: Interaction, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.print_events(logs)

    # same behaviour if we reached early stopping
    def on_early_stopping(self, _train_loss, _train_logs, epoch, _test_loss, test_logs):
        self.print_events(test_logs)
