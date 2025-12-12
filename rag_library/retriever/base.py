from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.Models import QueryWithResults


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> QueryWithResults:
        raise NotImplementedError
