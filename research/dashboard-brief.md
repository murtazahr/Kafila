# Design brief: split-inference operations dashboard

Paste this into Claude (or any design tool) to get a design back. Everything
below is real — the JSON shapes, the field names, and the numbers are copied
from a running three-node cluster, not invented. A design built on invented data
will not survive contact with the system.

---

## The prompt

> Design a single-page operations dashboard for a distributed LLM inference
> system. It sits beside a chat window and shows what happens to a prompt as it
> moves through a model that has been split across several machines.
>
> **Aesthetic:** a mission-control or signals-intelligence console. Dark,
> dense, monospaced, technical. It should look like an instrument rather than a
> marketing page — the kind of screen where every pixel is reporting something.
> No decorative illustration, no stock iconography, no gradients-for-their-own
> sake. Restraint is the point: one accent colour doing real work beats five
> doing none. It will be shown on a projector in a bright room, so contrast
> matters more than subtlety.
>
> **What the system is.** A language model is 28 transformer blocks. Rather than
> running all 28 on one machine, the blocks are divided across three processes,
> each on its own node. A prompt enters at the head node, which holds the token
> embedding and blocks 0–9. The head passes a hidden state to the next node,
> which runs blocks 10–18, which passes to the last, which runs blocks 19–27 and
> applies a final normalisation. The result travels back to the head, which
> turns it into a token and starts again for the next one. Every generated token
> makes this full circuit.
>
> **What must be legible at a glance, in priority order:**
>
> 1. **The topology** — which node holds which blocks, and whether each is
>    healthy. This is the spine of the screen.
> 2. **Live flow** — which node is working right now, and the direction of
>    travel. A viewer should see the circuit happening, not read about it.
> 3. **Where the time goes** — per-node compute versus time on the wire. This is
>    the whole point of the system: it answers "what does splitting cost?"
> 4. **Throughput and totals** — tokens per second, tokens generated, bytes
>    moved.
> 5. **Events** — cache alignment, resets, errors. Rare, but they explain
>    anomalies when they happen.
>
> **Design the following states**, since a dashboard that only looks good busy is
> not finished:
>
> - **Idle** — cluster up, no request in flight.
> - **Prefill** — the prompt is being processed in chunks. Payloads are large
>   (megabytes) and the nodes can overlap.
> - **Decode** — tokens generated one at a time. Payloads are tiny (kilobytes)
>   and the nodes strictly take turns; only one is ever busy.
> - **Degraded** — one node unreachable.
>
> The prefill/decode distinction is the most interesting thing on the screen and
> should be visibly different, not just a label that changes. In prefill the
> pipeline can be busy everywhere at once; in decode it is a baton passing round
> a ring with everyone else waiting.

---

## Real data the design must render

### Topology — `GET /v1/topology`

```json
{
  "model": "qwen3-mlx:0.6b",
  "blocks": 28,
  "tied": true,
  "stage_count": 3,
  "nodes": [
    {"index": 0, "name": "head",   "blocks": "[0,10)",  "role": "head",   "address": "local"},
    {"index": 1, "name": "stage1", "blocks": "[10,19)", "role": "middle", "address": "127.0.0.1:9301"},
    {"index": 2, "name": "stage2", "blocks": "[19,28)", "role": "tail",   "address": "127.0.0.1:9302"}
  ]
}
```

Note the roles are not symmetric, and the design should show why: the **head**
holds the embedding table, the output projection and the sampler as well as its
blocks, so it is doing markedly more than its block count suggests. The **tail**
holds the final norm. Middle nodes hold blocks only.

### A hop between nodes — one line of the NDJSON trace

```json
{"type":"hop","request":"pipeline","from":"head","to":"stage1","phase":"decode",
 "token":7,"chunk":-1,"round_trip_ns":6676000,"remote_duration_ns":6079000,
 "symmetric":true,"bytes":2048}
```

`round_trip_ns` is measured by the sender. `remote_duration_ns` is what the
receiver reported spending. **The difference is time actually in flight** — that
subtraction is the only honest way to separate network from compute here,
because the two machines do not share a clock. A design that shows round-trip
time as though it were network time would be actively misleading.

### Work on one node — the other line type

```json
{"request":"pipeline","node":"head","phase":"decode","kind":"compute",
 "token":7,"chunk":-1,"seq":12,"blocks":"[0,10)",
 "offset_ns":81234000,"duration_ns":10500000}
```

`kind` is one of `compute`, `serialize`, `deserialize`, `wire`, `queue`, `wait`.
`phase` is one of `load`, `align`, `prefill`, `decode`.

**There is deliberately no timestamp field.** Nodes do not share a clock, so
every measurement is a duration on one machine's own clock. Nothing in the
design may imply a synchronised global timeline — no wall-clock axis across
nodes. Ordering comes from token index and causality.

---

## Real measured numbers — use these, not placeholders

| Quantity | Value |
|---|---|
| Model | qwen3-mlx:0.6b, 28 blocks, tied embeddings |
| Split | 3 nodes: `[0,10)` `[10,19)` `[19,28)` |
| Throughput, split | ~47 tokens/sec |
| Throughput, single node | 94.94 tokens/sec |
| Decode payload per hop | **2048 bytes** |
| Prefill payload per chunk | **4 MiB** (2048 tokens × 1024 wide × bf16) |
| Round trip, decode hop | ~6.7 ms |
| Time genuinely in flight | ~550 µs |
| Transport floor, loopback | 125 µs for a 2 KiB round trip |
| Hops per 24-token request | 52 |
| Memory held: head | 507 MiB |
| Memory held: middle, tail | 210 MiB each |

The gap between the 6.7 ms round trip and the 550 µs in flight is the story:
**most of a hop is the next node computing, not the network.** If the design
makes that clear at a glance it has done its job.

---

## Constraints

- **One self-contained HTML page.** Inline CSS and JS, no external fonts,
  scripts or images. It is served by a Go process on a laptop.
- **Live updates over Server-Sent Events**, roughly one event per hop. During
  decode that is a few hundred per second, so the design must not require a
  full re-layout per event.
- **Readable on a projector.** Assume a bright room and a viewer three metres
  away for the headline numbers, closer for the detail.
- **Dark and light both**, since we do not control the room.
- **No invented data.** If a value is not in the tables above, the design should
  not show it. It is better to leave a panel out than to imply we measure
  something we do not.

## What to avoid

- Pie charts, gauges, and anything decorative that carries one number.
- A world map. The nodes are on one desk.
- Implying a synchronised clock across nodes.
- Presenting round-trip time as network latency.
- A layout that only reads well while traffic is flowing.

## Deliverable

A single HTML file, plus a short note on the reasoning: what earns its place on
the screen, what was deliberately left out, and how the prefill and decode
states differ visually.
