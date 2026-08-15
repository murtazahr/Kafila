# Open research questions

Building the split-inference backend is engineering. This file is where the
*research* problems it surfaces get logged, so they don't get silently resolved
with an expedient hack while the platform is being built, and don't get lost
either.

Each entry records what was observed, why it is a research problem rather than a
task, and what the platform currently does instead. The current behaviour is
deliberately the simplest thing that works — the point is to have a working
substrate to study these on, not to solve them early.

Status values: `open`, `parked` (waiting on the platform), `answered`.

---

## R1 — Decode cannot pipeline, and micro-batching is blocked

**Status:** open · **Surfaced:** designing the shard protocol

Prefill pipelines well: chunks are independent given the local KV cache, so a
node can start chunk *k+1* as soon as it finishes chunk *k*, without waiting for
downstream. Measured efficiency should approach `(K + N - 1) / K` for K chunks
across N nodes.

Decode does not pipeline at all. `pipelinedDecoder.next` needs token *N*'s
sampled value to dispatch token *N+1*, and with the ring topology the sampler
lives on the head, so every token costs a full traversal: N−1 forward hops plus
the backward edge. All stages idle while waiting.

The standard fix is cross-request micro-batching — keep every stage busy with
different requests. That is blocked at the cache layer:
`x/mlxrunner/cache/kvcache.go` states `Assumes B = 1; heterogeneous batches are
not supported`, and the runner serializes requests to a single pipeline slot.

**Why research:** lifting `B = 1` is not just a code change; heterogeneous
batching interacts with the prefix cache, the snapshot/restore machinery, and
per-slot sampler state. And even with it, the question of what scheduling
discipline minimizes p99 latency under pipeline parallelism is open. Speculative
cross-shard execution is another angle — the drafter already exists in
`x/mlxrunner/speculate.go`.

**Current behaviour:** decode is serial and pays N hops per token. Accepted.

---

## R2 — Distributed KV cache coherence

**Status:** open · **Surfaced:** reading `prefix_cache.go` invariants

`prefix_cache.go` states the invariant plainly: *all cache layers must stay at
the same token offset*. In one process a loop enforces it. Across processes
nothing does, and violating it is silent: `minCacheOffset()` feeds
`Batch.SeqOffsets`, which drives RoPE and mask construction, so divergent
offsets produce plausible-looking wrong output rather than an error.

Worse, not every cache can be re-aligned. `RotatingKVCache.Restore` refuses a
live rewind once the sliding window has wrapped, and `RecurrentCache.Restore`
can only succeed if already at the target — recurrent state is cumulative and
cannot rewind at all. For shards containing those layers the only fallback is a
global flush and full reprocess.

**Why research:** the cheap answer (coordinator computes a global minimum, every
node restores to it, divergence triggers a flush) is correct but wasteful, and
gets worse as node count grows. When is divergence actually tolerable? Can
non-rewindable cache kinds be checkpointed cheaply enough to avoid flushes? Is
there a formulation where nodes hold different offsets safely, with position
carried explicitly rather than derived?

**Current behaviour:** coordinator-owned alignment with a mandatory round-trip
before prefill; divergence is a global flush. Correct, not clever.

---

## R3 — Prefix cache divergence through independent eviction

**Status:** open · **Surfaced:** reading `cache_trie.go` eviction policy

`findBestMatch` is deterministic on the token stream, so per-node prefix tries
agree — until their eviction histories differ. Eviction is byte-driven against
an 8 GiB budget, and shards hold different-sized caches (attention vs Mamba
layers, different head counts, different block counts). Node A can evict an entry
node B still holds, after which the two compute different matches for the same
prompt and silently process different token ranges.

**Why research:** this is a distributed cache coherence problem with an unusual
cost asymmetry — a false miss costs recomputation, a false hit costs
correctness. Making the coordinator own all match decisions is safe but
serializes a hot path and discards per-node locality. Whether a weaker
consistency model is sound here is genuinely unclear.

**Current behaviour:** coordinator owns matching; nodes are told what to restore
to and never run `begin()` independently.

---

## R4 — Partitioning policy under heterogeneity

**Status:** parked (needs the platform) · **Surfaced:** measuring Qwen3-0.6B

The baseline planner splits blocks evenly. Real deployments break every
assumption behind that:

- **Non-uniform layer cost.** MoE layers, hybrid recurrent layers
  (`nemotron_h`, `qwen3_5`), and sliding-window layers have very different
  compute and memory profiles from dense attention blocks.
- **Non-uniform node capability.** Mixed Apple Silicon and CUDA, different
  memory ceilings, different interconnect quality per link.
- **Weight pinned to the ends.** The head holds the embedding, output
  projection and sampler regardless of how few blocks it owns.

On Qwen3-0.6B the last point is severe enough to make balance impossible: the
embedding is 297 MiB against 840 MiB of blocks, so it alone exceeds an equal
share at 4 shards. See R10.

**Why research:** this is a constrained optimization whose objective isn't even
settled — TTFT, sustained throughput, peak per-node memory, and energy point at
different partitions. It also interacts with R1, since a partition that is
optimal for prefill may be wrong for decode.

**Current behaviour:** even block split, capability-blind. Deliberately the
dumbest baseline so smarter planners have something to beat.

---

## R5 — Activation transport precision

**Status:** parked (needs measurement) · **Surfaced:** building the byte path

Activations cross the wire in the model's native dtype, bf16 today. That is
already half of what the F32 path would cost, but it is not obviously the floor.
Per-hop payload is `tokens × hidden_size × dtype_bytes`: 2 KiB per decode token
and 4 MiB per prefill chunk on Qwen3-0.6B, and roughly 4× that on a
4096-wide model.

**Why research:** activations are not weights — the error tolerance of a hidden
state passed between blocks is not the same as that of a quantized weight, and
it likely varies by depth. Whether int8 or a learned codec is viable without
measurable quality loss, and where in the network it stops being viable, is an
empirical question worth answering carefully.

**Current behaviour:** native dtype, no compression.

---

## R6 — Cut granularity

**Status:** parked (needs the platform) · **Surfaced:** design

Splitting at block boundaries is the obvious cut and the one with the smallest
interface: one hidden state per hop. Tensor-parallel cuts inside a layer trade
far more communication for lower per-stage latency, and hybrid schemes exist.

**Why research:** which cut wins is a function of interconnect bandwidth and
latency, model shape, and batch size, and the crossover points are not known for
Apple-Silicon-over-Ethernet or Thunderbolt clusters specifically. This is
exactly the sort of question the platform is being built to answer.

**Current behaviour:** block-boundary pipeline parallelism only.

---

## R7 — MoE expert placement

**Status:** parked (needs the platform) · **Surfaced:** surveying `x/models`

Several implemented architectures are MoE (`qwen3_5_moe`, `glm4_moe_lite`,
`cohere2_moe`). Ollama's import packs a layer's experts into a per-layer group
blob, so they shard by layer index today and the block-range split works
unmodified.

**Why research:** experts are the natural unit of a *different* partitioning
axis. Routing is dynamic and skewed, so expert placement is a load-balancing
problem with a data-dependent workload, quite unlike static layer ranges.

**Current behaviour:** experts follow their layer. No expert-level placement.

---

## R8 — Stragglers and failure

**Status:** open · **Surfaced:** design

A pipeline runs at the speed of its slowest stage, and a node lost mid-request
fails the request. Neither is addressed.

**Why research:** checkpointing every token is far too expensive; doing nothing
makes an N-node system N times more likely to fail a request. Where the useful
middle sits — and whether a shard's KV state can be reconstructed or migrated
cheaply — is open. Interacts with R2, since recovery is an alignment problem.

**Current behaviour:** none. A lost node fails the request.

---

## R9 — Does lazy loading hold on discrete VRAM?

**Status:** ANSWERED — yes, identically · **Surfaced:** Stage 0 experiment #1
· **Answered:** on an L40S (CUDA 12.8, driver 570), 2026-08-14

Re-running the same measurement on discrete VRAM reproduces the Metal result
to 0.1 MiB in every configuration. Resident memory after loading is 0.0 MiB
whatever is loaded, and after evaluation it tracks manifest bytes: 1433.6 MiB
for the whole model, 210.0 MiB for a quarter of it. The decisive comparison
holds too — an unfiltered load evaluating only one shard's 77 of 311 tensors
costs 210.0 MiB, exactly what the filtered load costs.

So the conclusion carries: **laziness, not the manifest filter, is what keeps
device memory proportional to what a node uses**, on unified and discrete
memory alike. The filter remains worth keeping for file work and for making
it impossible to bind a foreign block, but it is not the capacity mechanism
and the design should not claim it is.

The original entry follows.

---

Measured on Metal: MLX loads safetensors lazily, and resident memory after
loading is 0.0 MiB regardless of how much was loaded. An unfiltered load that
evaluates only one shard's tensors costs exactly the same resident memory as a
filtered one (210.0 MiB either way), so laziness — not the manifest filter — is
what keeps device memory proportional to what a node uses.

**Why it matters:** Apple Silicon has unified memory, where a lazy mmap-backed
tensor costs nothing until touched. A discrete GPU has to copy to device memory,
and the deferral may not survive. If it does not, the manifest filter changes
from an I/O optimization into the mechanism that makes sharding viable at all on
CUDA nodes.

Closer to verification than research, but it is load-bearing for the design and
must be re-measured before trusting any CUDA result.

**Current behaviour:** filter always applied, so correctness does not depend on
the answer.

---

## R10 — Embedding-dominated models

**Status:** open · **Surfaced:** measuring Qwen3-0.6B

Qwen3-0.6B has a 151,936-token vocabulary over a 1024-wide hidden state, so its
embedding is 297 MiB against 840 MiB of transformer blocks — 26% of model
weight in a single tensor that the ring topology pins to the head. At 4 shards
an equal share is 25%, so **no block assignment can balance the head**.

This is an artifact of small models with large vocabularies and inverts as
models grow. It matters for two reasons: results measured on this model will
make the head look worse than a realistic deployment, and it raises whether the
embedding should itself be shardable.

**Why research:** a vocabulary-sharded embedding is possible — each node holds a
slice of rows, and either the token is routed to the owning node or the slices
are gathered. Both cost a round trip on the critical path, so whether it beats
pinning the whole table to one node depends on interconnect and model shape.

**Current behaviour:** embedding pinned to the head, unsharded.

---

## R11 — Prefix cache reuse across a ring

A single node reuses KV across requests by rewinding its caches to the longest
prefix a previous request left behind. A ring cannot: the head can only rewind
the caches in its own process, and the other nodes go on appending.

The failure mode found this way is worth recording, because it is not a crash.
Rewinding the head while its peers do not leaves them attending over the
previous request's keys, and what comes back is fluent, confident, and about
whatever was asked previously. A prompt about one subject was answered on a
different one entirely, and nothing in any log looked wrong.

**Why research:** reuse across a ring means every node rewinding to the same
point, which needs an agreed prefix, a way to name it that survives all of them
independently evicting, and a rewind primitive on cache kinds that cannot all
rewind. Snapshot-and-restore per node has a memory cost that may exceed what the
reuse saves. Whether prefix reuse is worth its coordination cost in a split
deployment is open, and the answer probably depends on how much of a workload's
prompt is genuinely shared.

**Current behaviour:** a split model resets every node before every request, and
the runner drops its reuse trie to match. Correct, and it reprocesses prompts a
single node would not have.

---

## R12 — Where the chat template is applied

The MLX runner is a completion engine: it continues text and knows nothing about
turns. In the Ollama stack the chat template is applied above it, and for MLX
models that path is an unimplemented stub, so anything talking to the runner
directly gets raw continuation. An instruction-tuned model asked "Is X any
good?" carries on writing the document that question started rather than
answering it.

The console works around this by writing Qwen3's ChatML out by hand. That is
fine for one model family and wrong as a design: every caller reimplements the
same thing, and each gets a different model's format wrong differently.

**Why research:** less a research question than a design one, but it has a real
constraint attached — the template ships in the checkpoint's
tokenizer_config.json as Jinja, and the runner has no Jinja engine. Which layer
should own it, and whether a split deployment changes the answer (the head is
the only node that sees tokens at all), is worth settling before more callers
appear.

**Current behaviour:** ChatML hardcoded in the console.

---

## Answered

*(none yet)*
