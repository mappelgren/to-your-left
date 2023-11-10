from abc import ABC, abstractmethod

import numpy
import torch


class Persistable(ABC):
    @abstractmethod
    def save(self, file_path):
        ...

    @classmethod
    @abstractmethod
    def load_file(cls, file_path):
        ...


def load_tensor(data):
    match type(data):
        case numpy.ndarray:
            return torch.from_numpy(data)
        case _:
            return torch.tensor(data)
