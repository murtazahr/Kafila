# Kafila

A research platform for **distributed split inference**: one model cut at transformer
block boundaries, its pieces loaded onto separate machines, and a hidden state passed
around a ring until it comes back as a token.

Built on [Ollama](https://github.com/ollama/ollama), and deliberately invisible to it.
The split pipeline presents itself through the ordinary runner HTTP interface, so the
GUI, model downloads, the REST API and everything else work unchanged and unaware.
There is no distributed code path in the server and no flag threaded through the API —
the head node simply *is* a runner, from the outside.

> **Status: research, early.** The transport, serialisation and topology are what they
> would be across machines, but the reference deployment is three processes on one Mac
> over loopback. See [Limits](#limits).

## What it does

A transformer is an embedding, a stack of identical blocks, a final norm, and an output
projection. Block *k* consumes a hidden state and produces one of the same shape, so any
contiguous run of blocks has the same type signature as the whole stack — which makes
the block boundary the natural place to cut.

Twenty-eight blocks across three nodes:

```
head    blocks [0,10)   + embedding table, output projection, sampler
stage1  blocks [10,19)  blocks only
stage2  blocks [19,28)  + final norm
```

The head owns both ends of the model, so a frame that has been all the way round the
ring arrives back where it can be unembedded and sampled. Each node holds only the
tensors and KV caches for its own blocks.

Model files need no changes to be shardable. Tensors are renumbered on the way in — a
node holding blocks 10–19 reads `model.layers.11.*` and presents it as
`model.layers.1.*` — so an unmodified `LoadWeights` binds them without the model ever
learning its absolute position.

## Running a split cluster

Build, then start three processes:

```bash
go build -o ollama .
```

```bash
./scripts/split_cluster.sh start
```

Shards take about 40 seconds to load. The head then serves the runner interface *and*
the operations console on port 9300:

```bash
open http://127.0.0.1:9300
```

To see what distance does to a decode loop, inject one-way link delays standing in for
Melbourne → Mumbai → Sydney → Melbourne:

```bash
SIMULATE=1 ./scripts/split_cluster.sh start
```

```bash
./scripts/split_cluster.sh stop
```

Injected delay is applied on the wire and carried in its own field the whole way through
the trace, never summed into anything measured, so a demonstration cannot later be read
back as a result.

Ollama's own single-node commands are untouched:

```bash
./ollama serve
```

## Measuring across machines that do not share a clock

This is the part that makes the platform useful for research rather than only for demos.

NTP on commodity hardware leaves tens of milliseconds of skew, while a decode hop between
two nodes is sub-millisecond. Subtracting one node's timestamp from another's therefore
measures clock offset, not transit, and can produce negative transit times — figures that
are worthless and, worse, look plausible.

So **no span in the system carries an absolute timestamp**. Every measurement is a
duration on one node's monotonic clock, and cross-node ordering comes from causality —
token index, sequence number, the hop edges themselves — never from comparing clocks.

A circuit is attributed like this:

- the head times the **whole circuit** on its own clock
- each node measures **its own compute** on its own clock and appends it to the frame
- the frame arrives back carrying the circuit's complete account of itself
- circuit time minus the sum of reported compute is **time genuinely in flight**

Every subtraction is between durations. Where a one-way figure is derived by halving a
round trip, the symmetry assumption is recorded as a field rather than applied silently.

Under simulated inter-city latency, one decode circuit:

| segment | compute | injected | blocks |
|---|---|---|---|
| head | 11.0 ms | +75.0 ms | `[0,10)` |
| stage1 | 8.0 ms | +80.0 ms | `[10,19)` |
| stage2 | 7.2 ms | +7.0 ms | `[19,28)` |
| **circuit** | **26.1 ms** | **153.2 ms in flight** | **179.3 ms** |

Throughput falls from 57 tok/s to 5.2. That collapse is the point.

Prefill pipelines — chunk *n+1* can start at the head while chunk *n* is still
downstream. Decode cannot: token *n+1* needs token *n*'s sampled value, so exactly one
node is busy at a time. That asymmetry is the shape of the problem rather than an
implementation shortcoming, and it is what makes partitioning policy a real question.

## Layout

Everything new lives under `x/`, alongside the MLX runner it extends.

| Path | What it holds |
|---|---|
| `x/mlxrunner/shard/` | How a model divides: ranges, roles, tensor classification, key remapping |
| `x/cluster/agent/` | A node: shard loading, the ring links, the composed model the runner sees |
| `x/cluster/wire/` | Framed transport — JSON header plus raw tensor bytes |
| `x/cluster/trace/` | Spans, hops and the clock discipline above |
| `x/cluster/console/` | The operations dashboard |
| `x/mlxrunner/shardnode.go` | The `runner --shard` entrypoint every node runs |
| `scripts/split_cluster.sh` | Starts a local cluster |
| `research/` | Open questions, platform notes, design briefs |

Activations cross the wire as raw bytes in the model's native dtype. bf16 is read
through a `uint16` view rather than converted: cgo cannot represent `__bf16` at all, and
MLX omits 16-bit float accessors on CUDA builds, so asking for the float crashes there.
Reinterpreting the same width sidesteps both and is bit-identical.

## Research questions

Building the backend is not itself the research. Problems encountered along the way are
logged in [`research/open-questions.md`](research/open-questions.md) — twelve so far,
each recording what is open, why it is not merely an engineering task, and what the
system currently does instead. Among them:

- **R1** — decode cannot pipeline, and micro-batching is blocked at `B=1`
- **R2** — distributed KV cache coherence
- **R4** — partitioning policy under heterogeneous nodes
- **R5** — activation transport precision
- **R11** — prefix cache reuse across a ring

## Limits

Stated plainly, because each is a place a result could otherwise be over-read.

- **Three processes, one machine.** Communication is over loopback. One shard per
  process is not a demo convenience — two shards evaluating concurrently in one process
  deadlock inside MLX — so the local processes are treated exactly as separate machines
  would be. The network is the only thing that is not real.
- **MLX/Metal only.** CUDA nodes are supported in the code but not yet in a running
  cluster. CPU nodes are not implemented.
- **Simulated latency is injected, not incurred**, and is reported apart from every
  measurement.
- **No prefix cache reuse for split models.** Reuse across a ring means rewinding every
  node to a common point; until then a split model resets every node per request (R11).
- **No speculative decoding.** A draft head would have to run across the same split as
  the target, which is a separate problem.
- **The chat template is hardcoded** for one model family in the console, because the
  MLX runner does not implement the template path (R12).

## Building

```sh
cmake -B build .
cmake --build build --parallel 8
```

For Go-only iteration against an existing native payload:

```sh
go build .
```

See [`AGENTS.md`](AGENTS.md) for the development workflow and
[`docs/development.md`](docs/development.md) for prerequisites, platform notes and GPU
backends.

## Relationship to Ollama

Kafila is a fork of [ollama/ollama](https://github.com/ollama/ollama). Ollama at this
commit is not an inference engine: it resolves models, starts runner subprocesses, and
talks to them over HTTP. That leaves one seam worth extending — `llm.LlamaServer` — and
anything satisfying it is a valid backend as far as the rest of the system is concerned.
Kafila adds a backend that happens to span several machines.

Upstream capabilities — the CLI, the REST API, the model registry, GGUF models via
`llama.cpp` — are unmodified and documented at [docs.ollama.com](https://docs.ollama.com).

## License

Kafila is dual-licensed:

- **[PolyForm Noncommercial 1.0.0](LICENSE.md)** — free for research, teaching, study and
  personal use. If you are at a university or working on this for something other than
  commercial advantage, this covers you and there is nothing to sign.
- **Commercial** — for use in a product, a paid service, or commercial operations. See
  [`LICENSING.md`](LICENSING.md).

Kafila is a fork of Ollama and bundles llama.cpp, MLX and Leaflet. All are permissive and
permit redistribution under these terms with their notices retained; those notices are in
[`NOTICE`](NOTICE), and Ollama's MIT licence is kept verbatim in
[`LICENSE-OLLAMA-MIT`](LICENSE-OLLAMA-MIT). Nothing here limits any right you hold in that
upstream software under its own licence — it remains available from its authors under MIT.

Copyright is held by Murtaza Rangwala. Commercial enquiries: <murtazahatimr@icloud.com>.
