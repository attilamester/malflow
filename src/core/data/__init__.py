from abc import abstractmethod, ABC
from typing import Generator


class DatasetProvider(ABC):

    @abstractmethod
    def get_samples(self) -> Generator:
        pass
