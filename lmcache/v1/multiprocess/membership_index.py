# SPDX-License-Identifier: Apache-2.0
"""Server-side membership index for the worker-local residency mirror.

The MP scheduler adapter submits a blocking LOOKUP RPC for every new request
(`maybe_submit_lookup_request` waits on the job-id future). Under a cold flood
the server's queue is dominated by store traffic, each submit crawls, and the
vLLM scheduler convoys behind it (2026-07-17: 650k tok/s from a 4M-capable
box; 2026-07-20 phase-B rung-2 collapse). The fix is client-side: workers keep
an in-RAM mirror of which chunk hashes exist server-side and resolve COLD
misses to 0 instantly, submitting the RPC only when the mirror says a prefix
may exist.

This module is the authoritative source for that mirror:

- Subscribes as an ``L2AdapterListener`` (stored / deleted) so the set tracks
  the durable tier exactly; L1-only entries are deliberately ignored (they
  drain to L2 within seconds, and a false negative merely recomputes).
- Entries are ``cache_salt-scoped``: ``salt|chunk_hash`` — cross-salt
  collisions can never produce a false positive for another tenant.
- Deltas are kept in a bounded ring; a client whose epoch has fallen out of
  the ring gets a full snapshot instead.

Consistency model: the mirror is advisory. A stale-mirror false NEGATIVE loses
a cache hit (request recomputes — safe); a false POSITIVE costs one LOOKUP RPC
that reports 0 (safe). Correctness never depends on freshness.
"""

from dataclasses import dataclass, field
import threading

from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener

logger = init_logger(__name__)

# One ring entry per listener callback (batch), not per key — a callback
# carries every key of one store/eviction sweep. 4096 batches of history is
# hours of churn; clients sync every ~5s.
_RING_CAPACITY = 4096


def membership_entry(cache_salt: str, chunk_hash: bytes) -> bytes:
    """Wire/set representation of one membership entry."""
    return cache_salt.encode() + b"|" + chunk_hash


@dataclass
class _Delta:
    epoch: int
    added: list[bytes] = field(default_factory=list)
    removed: list[bytes] = field(default_factory=list)


class MembershipIndex(L2AdapterListener):
    """Chunk-hash membership set + epoch/delta log, fed by L2 events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: set[bytes] = set()
        self._epoch = 0
        self._ring: list[_Delta] = []

    # ---- L2AdapterListener ----------------------------------------------
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self._apply(keys, add=True)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self._apply(keys, add=False)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass  # access does not change membership

    # ---- internals -------------------------------------------------------
    def _apply(self, keys: list[ObjectKey], add: bool) -> None:
        # Dedup within the batch: one chunk yields one ObjectKey per
        # (kv_rank, object_group) — membership is per chunk, not per shard.
        entries = {membership_entry(k.cache_salt, k.chunk_hash) for k in keys}
        if not entries:
            return
        with self._lock:
            changed: list[bytes] = []
            if add:
                for e in entries:
                    if e not in self._entries:
                        self._entries.add(e)
                        changed.append(e)
            else:
                for e in entries:
                    if e in self._entries:
                        self._entries.discard(e)
                        changed.append(e)
            if not changed:
                return
            self._epoch += 1
            self._ring.append(
                _Delta(
                    epoch=self._epoch,
                    added=changed if add else [],
                    removed=[] if add else changed,
                )
            )
            if len(self._ring) > _RING_CAPACITY:
                del self._ring[: len(self._ring) - _RING_CAPACITY]

    # ---- sync API --------------------------------------------------------
    def sync(self, client_epoch: int) -> tuple[int, bool, list[bytes], list[bytes]]:
        """Return ``(epoch, is_snapshot, added, removed)`` for a client at
        ``client_epoch``. Snapshot when the client is new (epoch 0), ahead of
        us (server restart), or has fallen out of the delta ring."""
        with self._lock:
            if client_epoch == self._epoch:
                return self._epoch, False, [], []
            ring_start = self._ring[0].epoch if self._ring else self._epoch + 1
            if client_epoch == 0 or client_epoch > self._epoch or (
                client_epoch + 1 < ring_start
            ):
                return self._epoch, True, list(self._entries), []
            added: list[bytes] = []
            removed: list[bytes] = []
            for d in self._ring:
                if d.epoch > client_epoch:
                    added.extend(d.added)
                    removed.extend(d.removed)
            return self._epoch, False, added, removed

    def report_status(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "epoch": self._epoch}
