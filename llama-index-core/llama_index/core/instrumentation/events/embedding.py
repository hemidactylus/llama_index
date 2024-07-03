from typing import List

from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.schema import EmbeddingDeferred


class EmbeddingStartEvent(BaseEvent):
    """EmbeddingStartEvent.

    Args:
        model_dict (dict): Model dictionary containing details about the embedding model.
    """

    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingStartEvent"


class EmbeddingEndEvent(BaseEvent):
    """EmbeddingEndEvent.

    Args:
        chunks (List[str]): List of chunks.
        embeddings (List[List[float]]): List of embeddings.

    """

    chunks: List[str]
    embeddings: List[List[float] | EmbeddingDeferred]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingEndEvent"
