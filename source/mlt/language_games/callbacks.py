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

        # exclude space intesive information that is retrievable from the dataset
        logs.sender_input = torch.tensor(0)
        logs.receiver_input = torch.tensor(0)

        to_remove = ["masked_image", "caption", "train_mode", "attribute_tensor"]
        for key in to_remove:
            if key in logs.aux_input:
                logs.aux_input[key] = torch.tensor(0)

        torch.save(logs, dump_dir / f"interaction_gpu{rank}")
