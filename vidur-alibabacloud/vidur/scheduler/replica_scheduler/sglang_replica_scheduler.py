"""
SGLang replica scheduler for SimAI.

Simulates two key SGLang features:

1. **Chunked prefill** (similar to Sarathi-Serve)
   Long prompts are split into fixed-size chunks so that prefill and decode
   iterations can be interleaved, reducing head-of-line blocking.

2. **RadixAttention prefix caching**
   SGLang maintains a radix tree of KV-cache blocks indexed by token-prefix
   hashes.  When a new request shares a common prefix (e.g. a system prompt)
   with previously-cached content those KV blocks are reused without
   recomputation.

   Because this simulator works at the token-count level (not with actual token
   values) the prefix-cache benefit is approximated via a configurable
   ``prefix_cache_hit_rate``.  A hit rate of *r* means that the first
   ``floor(r * num_prefill_tokens)`` tokens of every new request are already
   present in the cache, so:

   * **Memory** – only ``ceil((1-r) * num_prefill_tokens / block_size)`` new
     KV blocks are allocated (the remainder are shared cache blocks).
   * **Scheduler latency** – the first scheduling iteration for that request
     advances ``num_processed_tokens`` by the full cached amount in one step
     (no extra attention computation is needed), reducing the number of
     prefill chunks required.

Usage example (via CLI)::

    python -m vidur.main \\
        --replica_scheduler_config_type sglang \\
        --sglang_scheduler_config_chunk_size 512 \\
        --sglang_scheduler_config_enable_prefix_caching True \\
        --sglang_scheduler_config_prefix_cache_hit_rate 0.7 \\
        --sglang_scheduler_config_max_tokens_in_batch 4096 \\
        ...
"""

from math import ceil
from typing import Dict, List, Set

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SglangReplicaScheduler(BaseReplicaScheduler):
    """SGLang-style replica scheduler with chunked prefill and prefix caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_running_batches: int = 0
        self._preempted_requests: List[Request] = []
        # Loose per-stage cap (memory is tracked explicitly via block allocation)
        self._max_micro_batch_size: int = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks: int = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

        # --- prefix-cache tracking ---
        # Maps request_id → number of prefill tokens satisfied by the prefix cache.
        # These tokens do not need new KV-block allocation.
        self._prefix_cache_hit_tokens: Dict[int, int] = {}
        # Set of request IDs whose cache-hit token "bump" has already been
        # included in a batch (so we don't add it twice).
        self._cache_hits_advanced: Set[int] = set()

    # ------------------------------------------------------------------
    # Prefix-cache helpers
    # ------------------------------------------------------------------

    def _get_cache_hit_tokens(self, request: Request) -> int:
        """Return the number of prefill tokens satisfied by the prefix cache."""
        if not self._config.enable_prefix_caching:
            return 0
        return int(request.num_prefill_tokens * self._config.prefix_cache_hit_rate)

    def _effective_new_prefill_tokens(self, request: Request) -> int:
        """Return the number of *new* (non-cached) prefill tokens for a request."""
        hit = self._prefix_cache_hit_tokens.get(
            request.id, self._get_cache_hit_tokens(request)
        )
        return max(0, request.num_prefill_tokens - hit)

    # ------------------------------------------------------------------
    # Memory allocation (overrides base class)
    # ------------------------------------------------------------------

    def _compute_required_blocks(self, effective_prefill_tokens: int) -> int:
        """Return the number of fresh KV blocks needed for *effective_prefill_tokens*.

        At least 1 block is always required so that the decode phase can track
        token capacity via ``_allocation_map``.
        """
        if effective_prefill_tokens > 0:
            return ceil(effective_prefill_tokens / self._config.block_size)
        return 1

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # New request: only non-cached tokens need fresh KV blocks.
            num_required_blocks = self._compute_required_blocks(
                self._effective_new_prefill_tokens(request)
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
        # Existing (decode-phase) request: needs room for at most 1 more block.
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # Compute and store cache-hit tokens for this request.
            hit = self._get_cache_hit_tokens(request)
            self._prefix_cache_hit_tokens[request.id] = hit
            effective = max(0, request.num_prefill_tokens - hit)
            # Always allocate at least 1 block so the decode phase can track
            # token capacity via _allocation_map.
            self.allocate(request.id, self._compute_required_blocks(effective))
            return

        # Decode phase: determine how many new blocks (if any) are needed.
        #
        # The "virtual" token capacity includes both physically allocated blocks
        # and cached tokens (which live in the shared prefix cache, not in new
        # blocks allocated to this request).
        hit = self._prefix_cache_hit_tokens.get(request.id, 0)
        num_tokens_available = (
            self._allocation_map[request.id] * self._config.block_size + hit
        )
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_available)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), (
            f"Expected decode-phase allocation delta of 0 or 1, "
            f"got {num_tokens_required} "
            f"(processed={request.num_processed_tokens}, "
            f"available={num_tokens_available})"
        )

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    # ------------------------------------------------------------------
    # Batch lifecycle callbacks
    # ------------------------------------------------------------------

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
                self._prefix_cache_hit_tokens.pop(request.id, None)
                self._cache_hits_advanced.discard(request.id)
            else:
                self._preempted_requests.append(request)

    # ------------------------------------------------------------------
    # Token-count helpers
    # ------------------------------------------------------------------

    def _get_request_next_num_tokens(
        self,
        request: Request,
        batch_contains_prefill: bool,
        num_batch_tokens: int,
    ) -> int:
        """Return how many tokens this request should process in the next chunk.

        For decode, this is always 1.

        For prefill the chunk budget is ``chunk_size - num_batch_tokens``.  On
        the *first* iteration of a prefix-cached request the cached-token bump
        is prepended so that ``num_processed_tokens`` jumps past the cached
        portion in one step; subsequent chunks proceed normally.
        """
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        # Determine whether this is the first time this request is being
        # scheduled (cache-hit bump not yet applied).
        cache_hit_bump = 0
        if (
            self._config.enable_prefix_caching
            and request.id in self._prefix_cache_hit_tokens
            and request.id not in self._cache_hits_advanced
        ):
            cache_hit_bump = self._prefix_cache_hit_tokens[request.id]

        # Remaining *new* tokens (excluding the not-yet-applied cache-hit bump).
        remaining_new = (
            request.num_prefill_tokens
            - request.num_processed_tokens
            - cache_hit_bump
        )

        next_new_tokens = min(
            max(0, remaining_new),
            max(0, self._config.chunk_size - num_batch_tokens),
        )

        total_tokens = cache_hit_bump + next_new_tokens
        if total_tokens == 0:
            return 0

        # Mark the cache-hit bump as consumed so it is not added again.
        if cache_hit_bump > 0:
            self._cache_hits_advanced.add(request.id)

        return total_tokens

    # ------------------------------------------------------------------
    # Core scheduling logic
    # ------------------------------------------------------------------

    def _restart_request(self, request: Request) -> None:
        """Evict a request, freeing its blocks and resetting prefix-cache state."""
        request.restart()
        self.free(request.id)
        # A restarted request gets a new (possibly different) num_prefill_tokens,
        # so discard stale prefix-cache bookkeeping.
        self._prefix_cache_hit_tokens.pop(request.id, None)
        self._cache_hits_advanced.discard(request.id)

    def _get_next_batch(self) -> Batch:  # noqa: C901
        requests: List[Request] = []
        num_tokens: List[int] = []
        skipped_requests: List[Request] = []
        running_prefills: List[Request] = []
        contains_prefill = False
        num_batch_tokens = 0

        # ----------------------------------------------------------------
        # 1. Process preempted requests (may include partial prefills)
        # ----------------------------------------------------------------
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                # Still in prefill phase – handle separately below.
                running_prefills.append(request)
                continue

            # Decode-phase preempted request.
            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            # Ensure there is enough memory; evict the youngest preempted
            # request if necessary.
            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim = self._preempted_requests.pop(-1)
                    self._restart_request(victim)
                    self._request_queue = [victim] + self._request_queue
                else:
                    self._restart_request(request)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        # ----------------------------------------------------------------
        # 2. Continue in-flight partial prefills
        # ----------------------------------------------------------------
        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # Restore skipped requests at the front (preserve FIFO ordering).
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda r: r.arrived_at
        )

        # ----------------------------------------------------------------
        # 3. Admit new requests from the queue
        # ----------------------------------------------------------------
        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break
            if len(requests) == self._max_micro_batch_size:
                break
            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )
            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)
            self._allocate_request(request)
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)
