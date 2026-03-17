"""Integration tests — index the Alexandria codebase and search it.

These tests exercise the full pipeline: file discovery, chunking, embedding,
storage, and search.  They require the isolated Qdrant and Ollama started by
``scripts/integration-test.sh``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from alexandria.chunker import Chunk, chunk_file
from alexandria.config import Config
from alexandria.discovery import discover_files
from alexandria.embedder import Embedder
from alexandria.store import Store
from tests.conftest import TEST_CONTEXT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_codebase(
    repo_root: Path,
    config: Config,
    store: Store,
    embedder: Embedder,
) -> int:
    """Run the same index pipeline as ``alex index`` and return chunk count.

    This mirrors the three-phase logic in cli.py but uses the test
    config so it talks to the isolated services.
    """
    files = discover_files(repo_root, follow_symlinks=False)
    assert len(files) > 0, "discover_files returned nothing"

    # Phase 1: chunk
    all_chunks: list[Chunk] = []
    for f in files:
        chunks = chunk_file(f, config, repo_root)
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0, "No chunks produced from the codebase"

    # Phase 2: embed
    texts = [c.text for c in all_chunks]
    vectors: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = embedder.embed_batch(batch, batch_size=len(batch))
        vectors.extend(batch_vectors)

    assert len(vectors) == len(all_chunks)

    # Phase 3: store
    n_stored = store.upsert_chunks(TEST_CONTEXT, all_chunks, vectors)
    return n_stored


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndexing:
    """Verify that we can index the Alexandria codebase into Qdrant."""

    def test_discover_files(self, repo_root: Path) -> None:
        """File discovery should find the source files in this repo."""
        files = discover_files(repo_root, follow_symlinks=False)
        paths = {str(f.relative_to(repo_root)) for f in files}
        # Expect at least the core source files
        for expected in [
            "src/alexandria/cli.py",
            "src/alexandria/config.py",
            "src/alexandria/embedder.py",
            "src/alexandria/store.py",
            "src/alexandria/chunker.py",
        ]:
            assert expected in paths, f"Expected {expected} in discovered files"

    def test_chunk_file(self, repo_root: Path, test_config: Config) -> None:
        """Chunking a known Python file should produce AST-aware chunks."""
        target = repo_root / "src" / "alexandria" / "config.py"
        chunks = chunk_file(target, test_config, repo_root)
        assert len(chunks) > 0, "config.py should produce at least one chunk"
        # Every chunk should reference the correct file
        for c in chunks:
            assert c.file == "src/alexandria/config.py"
            assert c.language == "python"

    def test_embed_single(self, test_embedder: Embedder) -> None:
        """Embedding a short text should return a 768-dim vector."""
        vec = test_embedder.embed("def hello(): pass")
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_full_index(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Index the entire Alexandria codebase and verify chunks are stored."""
        n_stored = _index_codebase(repo_root, test_config, test_store, test_embedder)
        assert n_stored > 0, "Expected at least one chunk stored"

        # Verify via list_contexts
        contexts = test_store.list_contexts()
        assert TEST_CONTEXT in contexts

        # Verify point count matches
        stats = test_store.get_context_stats(TEST_CONTEXT)
        assert stats["points"] == n_stored

    def test_incremental_index_skips_unchanged(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """A second index run should detect all files as unchanged."""
        # First index
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        # Check stored hashes
        hashes = test_store.get_indexed_file_hashes(TEST_CONTEXT)
        assert len(hashes) > 0

        # Verify hashes match actual file contents
        files = discover_files(repo_root, follow_symlinks=False)
        matched = 0
        for f in files:
            rel = str(f.relative_to(repo_root))
            if rel in hashes:
                actual_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                assert hashes[rel] == actual_hash, f"Hash mismatch for {rel}"
                matched += 1
        assert matched > 0, "No file hashes matched — index may have failed"


class TestSearch:
    """Verify that search returns relevant results after indexing."""

    def test_search_finds_embedder(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Searching for 'embedding' should surface embedder.py code."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        query_vec = test_embedder.embed("generate vector embeddings from text using Ollama")
        results = test_store.search(
            context=TEST_CONTEXT,
            query_vector=query_vec,
            limit=5,
        )
        assert len(results) > 0, "Search returned no results"

        # At least one result should be from embedder.py
        files_found = {r.file for r in results}
        assert "src/alexandria/embedder.py" in files_found, (
            f"Expected embedder.py in results, got: {files_found}"
        )

    def test_search_finds_store(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Searching for 'qdrant upsert' with python filter should surface store.py."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        query_vec = test_embedder.embed("QdrantClient upsert_chunks collection points")
        results = test_store.search(
            context=TEST_CONTEXT,
            query_vector=query_vec,
            limit=5,
            language_filter="python",
        )
        assert len(results) > 0

        files_found = {r.file for r in results}
        assert "src/alexandria/store.py" in files_found, (
            f"Expected store.py in results, got: {files_found}"
        )

    def test_search_with_language_filter(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Language filter should restrict results to that language."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        query_vec = test_embedder.embed("configuration")
        results = test_store.search(
            context=TEST_CONTEXT,
            query_vector=query_vec,
            limit=10,
            language_filter="python",
        )
        for r in results:
            assert r.language == "python", f"Expected python, got {r.language}"

    def test_search_all_across_contexts(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """search_all should find results across contexts."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        query_vec = test_embedder.embed("CLI command entry point")
        results = test_store.search_all(query_vector=query_vec, limit=5)
        assert len(results) > 0, "search_all returned no results"

    def test_search_result_has_score(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Every search result should have a positive similarity score."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)

        query_vec = test_embedder.embed("file discovery")
        results = test_store.search(
            context=TEST_CONTEXT,
            query_vector=query_vec,
            limit=3,
        )
        assert len(results) > 0
        for r in results:
            assert r.score > 0, f"Expected positive score, got {r.score}"

    def test_drop_context(
        self,
        repo_root: Path,
        test_config: Config,
        test_store: Store,
        test_embedder: Embedder,
    ) -> None:
        """Dropping a context should remove it entirely."""
        _index_codebase(repo_root, test_config, test_store, test_embedder)
        assert TEST_CONTEXT in test_store.list_contexts()

        test_store.drop_context(TEST_CONTEXT)
        assert TEST_CONTEXT not in test_store.list_contexts()
