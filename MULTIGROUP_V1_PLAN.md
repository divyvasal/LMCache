# engine_driven multi-group v1 — implementation plan (uniform coverage)

Working branch: `feat/engine-driven-multigroup` (base upstream/dev). Tracking issue:
LMCache#4071. Delete this file before opening the PR; its content becomes the PR body.

## Scope

v1 = uniform-coverage groups: every LMCache KV group stores/retrieves every chunk.
Correct for hybrids whose groups are all full-length (GLM/DeepSeek IndexShare-style
indexer groups). Groups with `sw_size_tokens >= 0` (sliding windows, e.g. gpt-oss)
are still rejected — with a clearer error — until v2 per-group coverage lands.
Pickle transport stays single-group (multi-group requires SHM mode; reject with a
clear error otherwise).

## Stages

### 1. Payload + worker register
- `RegisterEngineDrivenContextPayload` (custom_types.py) gains optional
  `engine_group_infos: list[EngineGroupInfo] = []` and
  `group_layouts: list[GroupLayout] = []` where GroupLayout is a small msgspec
  struct `{num_layers, hidden_dim_size, dtype_str}` (per LMCache group, order =
  protocol group order). Back-compat: empty lists = legacy single-group.
- Worker (`worker_transfer.EngineDrivenTransferContext.register`): when
  `engine_group_infos` non-empty and >1 group:
  - reject any group with `sw_size_tokens >= 0` (v1 limitation, clear message);
  - reject pickle mode;
  - compute per-group layout via `compute_kv_layout` on the kv_caches subset
    selected by `group.layer_indices` (keep dict order; layer_indices index the
    registered tensor order);
  - require all groups share `tokens_per_block` == engine block_size for v1?
    NO — support per-group `blocks_in_chunk_g = chunk_tokens / tokens_per_block_g`;
    fall back to detected block_size when `tokens_per_block == 0`.
  - store per-group `(kv_caches_subset, layout, engine_kv_format, blocks_in_chunk_g)`
    on self; single-group path unchanged.

### 2. Server register + key resolution
- `engine_driven_transfer.register_kv_cache_engine_driven_context`: when payload
  carries group_layouts, build `layouts: list[MemoryLayoutDesc]` (one per group,
  shape `[2, num_layers_g, chunk, hidden_g]` / MLA variant per use_mla) and store
  on the context entry (extend `EngineDrivenContextMetadata` with
  `group_layouts: list[MemoryLayoutDesc] | None = None`).
- `_resolve_single_group_obj_keys` → `_resolve_obj_keys_by_group(key)`:
  `self._ctx.resolve_obj_keys(key, list(range(n_groups)))` → `list[list[ObjectKey]]`
  (group-major). Single-group callers keep `[0]`.
- layout_desc_registry: register the group-0 layout as today (compat for lookup
  paths); add per-group registration only if a lookup path needs it (audit:
  prefetch/lookup use layout from registry keyed (model, world_size) — retrieve
  path in v1 reads layouts from the entry, not the registry).

### 3. Store path
- `ShmTransferStrategy.prepare_store`: for each group g: reserve_write(group_keys_g,
  layout_g); slots response becomes `{"slots": [...], "chunk_indices": [...],
  "group_ids": [...]}` — flat lists, parallel arrays, group-major within chunk
  order. `chunk_indices` shared: a chunk needs writing if missing in ANY group;
  for groups where the chunk exists, skip reserving (mode="new" already skips) —
  v1 simplification: reserve mode="new" per group, slot set = union; worker writes
  what it got slots for. pending_writes entry holds ALL reserved keys (flat).
- Worker `submit_store`: build per-group slot-tensor lists from the tagged response;
  for each group: gather(kv_subset_g, block_ids[g], blocks_in_chunk_g, layout_g,
  out=slots_g, chunk_indices=chunk_indices_g). Commit unchanged (b"" payload).
- Async context: same via the shared helpers (it reuses base register + calls the
  same gather; extend its Phase-2 to loop groups).

### 4. Retrieve path
- Server `prepare_retrieve`: unsafe_read per group; miss if ANY group incomplete
  (release partials); slots tagged by group as in store.
- Worker `submit_retrieve`: per-group scatter(kv_subset_g, block_ids[g], src_g,
  blocks_in_chunk_g, skip_first_n_tokens, layout_g).

### 5. Tests
- Unit: register payload round-trip (msgspec encode/decode with groups);
  worker register builds per-group layouts (2 fake groups, distinct hidden dims);
  sw-group rejection message; pickle+multi-group rejection.
- Server: prepare_store reserves per-group keys with per-group layouts (mock SM,
  assert reserve_write called per group with right layout + object_group_id);
  retrieve miss when one group absent.
- E2E-ish (existing pattern in test_engine_driven_transfer.py with real SHM pool):
  2-group store+retrieve round-trip, assert both groups' bytes land and scatter
  restores them.
- Livebox: full suites + glm validation (below).

### 6. Fleet validation (after upstream-shaped code is green)
- Build dev11 wheel (dev10 base + this diff, same in-place wheel-patch flow as
  dev10; bump `_version.py` + METADATA + RECORD).
- Box #6: REMOVE `--disable-hybrid-kv-cache-manager` for glm only
  (coral-model-swap-container.sh case split: glm drops the flag, gpt keeps it),
  full reassembly (NOT in-place recreate — bug #28 rule), glm gauntlet: boot,
  N salted stores, Stored lines, park/wake, post-wake infer, S3 objects with
  object_group_id in key names.

## Key file map
- lmcache/v1/multiprocess/custom_types.py — payload structs
- lmcache/v1/multiprocess/group_view.py — EngineGroupInfo (exists; reuse)
- lmcache/v1/multiprocess/transfer_context/base.py — EngineDrivenContextMetadata,
  compute_kv_layout, gather/scatter helpers
- lmcache/v1/multiprocess/transfer_context/worker_transfer.py — register +
  submit_store/submit_retrieve (sync), `_single_group_block_ids` (its rejection
  branches to the sw/pickle-specific messages)
- lmcache/v1/multiprocess/transfer_context/async_engine_driven.py — async store
- lmcache/v1/multiprocess/modules/engine_driven_transfer.py — server register +
  resolve + prepare/commit handlers
- lmcache/v1/multiprocess/modules/server_transfer.py — ShmTransferStrategy
- ctx.resolve_obj_keys(key, group_ids) — server-side existing multi-group key API

## Open questions posted upstream (#4071)
Per-group keys vs multi-tensor objects (going with per-group keys, matching
lmcache_driven); payload extension vs registration-path convergence; alignment
with the tokens_per_block TODO. If maintainers answer differently mid-flight,
adjust before PR.
