import os
import time
from typing import List, Dict, Any, Optional

from app.rag_engine import (
    EmbeddingRateLimitConfig,
    is_rate_limit_error,
    extract_retry_after,
    EmbeddingCacheStore,
    GoogleEmbeddingThrottler,
    RateLimitedGoogleEmbeddings,
)
from langchain_core.embeddings import Embeddings


class MockClientError(Exception):
    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status


def test_is_rate_limit_error():
    # Direct status 429
    err1 = MockClientError("Resource has been exhausted (e.g. check quota)", status=429)
    assert is_rate_limit_error(err1) is True

    # Nested exception with cause
    err2 = RuntimeError("Wrapped error")
    err2.__cause__ = err1
    assert is_rate_limit_error(err2) is True

    # Text-based indicators
    err3 = Exception("RateLimitError: 429 Too Many Requests")
    assert is_rate_limit_error(err3) is True

    err4 = Exception("GoogleGenerativeAIError: ResourceExhausted: Quota exceeded for project")
    assert is_rate_limit_error(err4) is True

    # Non-429 error
    err5 = ValueError("Invalid API key")
    assert is_rate_limit_error(err5) is False


def test_extract_retry_after():
    err1 = Exception("Rate limit exceeded. Please retry in 18.5s.")
    assert extract_retry_after(err1) == 18.5

    err2 = Exception("ResourceExhausted: 429. Retry-After: 30")
    assert extract_retry_after(err2) == 30.0

    err3 = Exception("Quota exceeded. Please wait 12 seconds before retrying.")
    assert extract_retry_after(err3) == 12.0

    err4 = Exception("ResourceExhausted: 429 Resource has been exhausted")
    assert extract_retry_after(err4) is None


def test_embedding_cache_store(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    cache = EmbeddingCacheStore(db_path=db_path)

    texts = ["apple", "banana", "cherry"]
    model = "models/gemini-embedding-001"

    # Initially empty
    res1 = cache.get_batch(texts, model)
    assert len(res1) == 0

    # Save mock vectors
    items = [
        (cache.hash_text("apple"), model, [0.1, 0.2, 0.3], 5),
        (cache.hash_text("banana"), model, [0.4, 0.5, 0.6], 6),
    ]
    cache.save_batch(items)

    # Lookup batch
    res2 = cache.get_batch(texts, model)
    assert len(res2) == 2
    apple_hash = cache.hash_text("apple")
    assert res2[apple_hash] == [0.1, 0.2, 0.3]

    # Check stats
    stats = cache.get_stats()
    assert stats["total_cached"] == 2
    assert stats["models"] == 1

    # Clear
    cache.clear()
    assert cache.get_stats()["total_cached"] == 0


def test_throttler_pacing_and_wait():
    config = EmbeddingRateLimitConfig(
        preset="custom",
        batch_size=5,
        inter_batch_delay=0.05,
        rpm_limit=3,
        tpm_limit=500,
        max_retries=3,
        initial_backoff=0.1,
        max_backoff=0.5,
    )
    throttler = GoogleEmbeddingThrottler(config)

    # First call: no wait
    w1 = throttler.wait_if_needed(estimated_tokens=50)
    assert w1 == 0.0

    # Second call immediately: waits inter_batch_delay (~0.05s)
    w2 = throttler.wait_if_needed(estimated_tokens=50)
    assert w2 > 0.0


def test_throttler_429_handle():
    config = EmbeddingRateLimitConfig(
        preset="custom",
        batch_size=5,
        inter_batch_delay=0.01,
        rpm_limit=10,
        tpm_limit=1000,
        max_retries=3,
        initial_backoff=0.05,
        max_backoff=0.2,
    )
    throttler = GoogleEmbeddingThrottler(config)

    notifications = []
    def on_wait(msg: str, wait_s: float):
        notifications.append((msg, wait_s))

    err = MockClientError("ResourceExhausted: 429", status=429)
    wait_time = throttler.handle_429(attempt=0, error=err, on_wait=on_wait)
    assert wait_time > 0.0
    assert len(notifications) == 1
    assert "429" in notifications[0][0]


class MockEmbeddingClient(Embeddings):
    def __init__(self, fail_on_batch: Optional[int] = None):
        self.call_count = 0
        self.fail_on_batch = fail_on_batch

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        self.call_count += 1
        if self.fail_on_batch and self.call_count == self.fail_on_batch:
            raise MockClientError("Resource has been exhausted (429)", status=429)
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 8


def test_rate_limited_embeddings_wait_resume(tmp_path):
    db_path = str(tmp_path / "test_rl.db")
    cache = EmbeddingCacheStore(db_path=db_path)

    config = EmbeddingRateLimitConfig(
        preset="custom",
        batch_size=2,
        inter_batch_delay=0.01,
        rpm_limit=20,
        tpm_limit=10000,
        max_retries=3,
        initial_backoff=0.05,
        max_backoff=0.2,
        enable_cache=True,
    )

    # Mock client will fail on batch 2 with 429, then succeed on retry
    mock_client = MockEmbeddingClient(fail_on_batch=2)

    progress_events = []
    def on_progress(evt: Dict[str, Any]):
        progress_events.append(evt)

    wrapper = RateLimitedGoogleEmbeddings(
        underlying=mock_client,
        model_name="models/gemini-embedding-001",
        config=config,
        cache=cache,
        progress_callback=on_progress,
    )

    texts = [f"document chunk {i}" for i in range(6)]
    vectors = wrapper.embed_documents(texts)

    assert len(vectors) == 6
    assert len(vectors[0]) == 8

    # Verify 429 was caught, wait was triggered, and resumed
    phases = [e.get("phase") for e in progress_events]
    assert "rate_limit_wait" in phases
    assert "resuming" in phases

    # Verify persistent cache: subsequent call with same texts should use 100% cached items
    mock_client2 = MockEmbeddingClient()
    wrapper2 = RateLimitedGoogleEmbeddings(
        underlying=mock_client2,
        model_name="models/gemini-embedding-001",
        config=config,
        cache=cache,
    )
    cached_vectors = wrapper2.embed_documents(texts)
    assert len(cached_vectors) == 6
    # mock_client2 should have 0 calls because all chunks were in cache!
    assert mock_client2.call_count == 0
