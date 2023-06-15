import json
import os
import pprint

from egg.core import Callback
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
