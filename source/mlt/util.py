from abc import ABC, abstractmethod


class Persistable(ABC):
    @abstractmethod
    def save(self, file_path):
        ...

    @classmethod
    @abstractmethod
    def load_file(cls, file_path):
        ...
