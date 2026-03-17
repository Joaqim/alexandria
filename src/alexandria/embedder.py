"""Embedder — generates vector embeddings from text chunks via Ollama."""

from __future__ import annotations

import ollama as ollama_client

from alexandria.config import Config


class Embedder:
    """Generates embeddings using Ollama's embedding API."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = ollama_client.Client(host=config.ollama_host)
        self.model = config.embed_model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector."""
        response = self.client.embed(model=self.model, input=text)
        # ollama returns {"embeddings": [[...]]} for embed()
        embeddings = response.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise RuntimeError(f"Ollama returned no embeddings for model {self.model}")

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors.

        Sends texts to Ollama in sub-batches using the native batch API
        (``input=list[str]``).  This is dramatically faster than
        one-at-a-time because each HTTP round-trip embeds up to
        *batch_size* texts in a single model invocation.

        Args:
            texts: The texts to embed.
            batch_size: Max texts per Ollama request.  Defaults to 64.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embed(model=self.model, input=batch)
            embeddings = response.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings "
                    f"for a batch of {len(batch)} texts (offset {i})"
                )
            all_embeddings.extend(embeddings)
        return all_embeddings

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is pulled."""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            # Model names may include tags like ":latest"
            return any(
                self.model in name or name.startswith(f"{self.model}:") for name in model_names
            )
        except Exception:
            return False

    def pull_model(self) -> None:
        """Pull the embedding model if not already available."""
        self.client.pull(self.model)
