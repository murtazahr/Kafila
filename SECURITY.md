# Security

## Reporting a vulnerability

Please do not open a public issue. Email <murtazahatimr@icloud.com> with:

- a description of the vulnerability
- steps to reproduce it
- your assessment of the impact
- any mitigations you can see

Kafila is a research project maintained by one person, so please allow
reasonable time to investigate before disclosing publicly.

**For vulnerabilities in upstream Ollama** unrelated to split inference, report
to <hello@ollama.com> instead, where a fix reaches everyone rather than this
fork alone.

## The cluster transport is unauthenticated

This is the most important thing to know before running Kafila anywhere other
than a trusted machine, and it is by design rather than by oversight.

Nodes talk to each other over **plain TCP with no authentication, no encryption,
and no verification of who is connecting**. A node accepts any well-formed frame
that arrives on its port and runs it through its share of the model.

Consequently, anyone who can reach a node's port can:

- submit activations and receive the resulting hidden state back
- reset that node's KV cache, disrupting a request in progress
- infer the model's shape and this node's block range from the frames

And anyone who can observe the traffic can read activations in the clear.
**Activations are not opaque** — a hidden state carries a great deal about the
input that produced it, and recovering text from one is an active research area
rather than a hypothetical. Treat the link between nodes as carrying the same
sensitivity as the prompt itself.

There is no threat model in which the current transport is safe on an untrusted
network. It is appropriate for loopback and for a private network you control.

If you need to run across a network you do not trust, put the links inside a
tunnel you do — WireGuard, an SSH tunnel, or a VPN — and bind nodes to the tunnel
interface rather than to a public one. Authentication and transport encryption
in the protocol itself are not implemented.

## Binding and exposure

`scripts/split_cluster.sh` binds every node to `127.0.0.1`, which is why the
reference setup is safe to run. Nodes are reachable beyond the machine only if
you pass a routable address to `--listen`, `--next` or `--return-listen`.

The head also serves the operations console and the runner HTTP interface on the
same port. The console has no authentication and shows the model, topology,
prompts and completions passing through the cluster. Exposing that port exposes
all of it.

## Scope

In scope for reports here:

- `x/cluster/` — the ring, the wire format, the trace recorder, the console
- `x/mlxrunner/shard/` and the shard node entrypoint
- `scripts/split_cluster.sh`

Out of scope, and better reported upstream:

- Ollama's server, API, model registry and GGUF path
- llama.cpp, MLX, and other bundled dependencies

The unauthenticated transport described above is known and documented, so it
does not need reporting — though a way to reach a node *beyond* what is described
there does.

## Running it safely

- Keep nodes on loopback or inside a network you control.
- Do not expose the head's port; the console and the runner interface both sit
  behind it with no authentication.
- Treat inter-node links as carrying prompt-sensitive data.
- Remember that traces written with `--trace` record request structure and
  timings, and that the console streams them to anyone who can reach it.
