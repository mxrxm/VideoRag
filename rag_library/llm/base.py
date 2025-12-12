from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a continuation given a prompt.
        """
        raise NotImplementedError
