import builtins
import inspect
import os
import pickle
from dataclasses import fields

import h5py
import numpy
import torch
from mlt.preexperiments.models import CaptionDecoder


def load_tensor(data):
    match type(data):
        case torch.Tensor:
            return data
        case numpy.ndarray:
            return torch.from_numpy(data)
        case builtins.list:
            match type(data[0]):
                case builtins.str:
                    return data
                case _:
                    return torch.stack([load_tensor(d) for d in data])
        case _:
            return torch.tensor(data)


class Persistor:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def save(self, samples, **kwargs):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with h5py.File(self.file_path, "w") as f:
                sample_fields = fields(samples[0])

                for field in sample_fields:
                    data = load_tensor(
                        [getattr(sample, field.name) for sample in samples]
                    )

                    f.create_dataset(
                        field.name,
                        data=data,
                    )

                for attribute, value in kwargs.items():
                    f.attrs[attribute] = numpy.void(pickle.dumps(value))
        except Exception as e:
            os.remove(self.file_path)
            raise e

    def load(self, cls):
        with h5py.File(self.file_path, "r") as f:
            # pylint: disable-next=no-member
            attributes = {
                attr: pickle.loads(attr_value.tobytes())
                for attr, attr_value in f.attrs.items()
            }
            return cls(self.file_path, **attributes)


class colors:
    GREEN = "\033[38;5;28m"
    RED = "\033[38;5;88m"
    ENDC = "\033[0m"


def get_model_params(model, opts, prefix=""):
    params = {}

    signature = inspect.signature(model.__init__)
    for name, par in signature.parameters.items():
        if name not in ["self", "_args", "_kwargs"]:
            match par.annotation.__qualname__:
                case CaptionDecoder.__qualname__:
                    params = params | get_model_params(CaptionDecoder, opts, prefix)
                case _:
                    if not name.startswith(prefix):
                        name = prefix + name
                    if name in opts:
                        params[name] = getattr(opts, name)

    return params


def set_model_params(model_args, params):
    for param, param_value in params.items():
        model_args[param] = param_value

    return model_args
