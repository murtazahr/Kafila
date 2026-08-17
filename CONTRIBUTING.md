# Contributing to Kafila

Kafila is a research platform for distributed split inference. It is early, and
the parts most worth improving are the ones the research questions point at
rather than the surface.

## Before you start: the contributor agreement

Kafila is dual-licensed — [PolyForm Noncommercial](LICENSE.md) for research and
personal use, commercial terms otherwise. Offering commercial terms requires the
licensor to hold rights to all the code being licensed, so **contributions need
a contributor licence agreement granting a licence broad enough to permit
commercial sublicensing**.

This has to be in place before a contribution is merged, not after. Please
[open an issue](https://github.com/murtazahr/Kafila/issues) or email
<murtazahatimr@icloud.com> before starting work, and the agreement can be sorted
out first. See [LICENSING.md](LICENSING.md).

## Set up

```sh
cmake -B build .
cmake --build build --parallel 8
```

For Go-only iteration against an existing native payload:

```sh
go build .
```

See [`AGENTS.md`](AGENTS.md) and [`docs/development.md`](docs/development.md) for
prerequisites, platform notes and GPU backends.

To run a split cluster locally:

```sh
go build -o ollama . && ./scripts/split_cluster.sh start
```

Three processes, one shard each, with the operations console on
<http://127.0.0.1:9300>.

## Where help is most useful

[`research/open-questions.md`](research/open-questions.md) records twelve open
problems, each with what is unresolved and what the system currently does
instead. Those are the substance of the project. R4 (partitioning under
heterogeneous nodes), R5 (activation transport precision) and R11 (prefix cache
reuse across a ring) are the ones where an implementation would settle an
argument.

Beyond those:

- **A second backend.** CUDA nodes work in code but have never run in a cluster.
  CPU nodes are not implemented at all.
- **More architectures.** Only Qwen3 implements `base.Sharded`. Most models
  should need very little — see `x/models/qwen3/qwen3.go`.
- **Measurement.** Anything that makes a figure more honest, or that catches a
  figure which is quietly wrong.

## What is harder to accept

- **Changes that break the upstream interface.** Kafila's whole approach depends
  on the split pipeline being indistinguishable from an ordinary runner. A change
  that requires the server to know about distribution defeats the point.
- **Cross-clock timing.** Nothing may derive a duration by subtracting one
  machine's timestamp from another's. See the package comment in
  `x/cluster/trace`. If a measurement seems to need it, that is the discussion to
  have in an issue first.
- **Mixing simulated and measured values.** Injected latency travels in its own
  field, all the way through. It must never be summed into anything measured.
- **Large refactors of upstream Ollama code.** They make merging upstream changes
  painful, and the benefit rarely covers it.

## Pull requests

**Commit messages.** The title looks like:

```
<package>: <short description>
```

The package is the most affected Go package; if the change does not touch Go
code, use the directory name. The description starts lowercase and continues the
sentence *"This changes Kafila to…"*:

```
x/cluster/agent: report resident memory from every node
research: record why prefix reuse cannot span a ring
```

Not:

```
feat: add more emoji
fix: various improvements
```

The body should explain **why**, not what — the diff already says what. If a
change encodes a non-obvious constraint, say so there; several of this
repository's sharpest bugs were silent, and the commit message is where the next
person finds out why the code looks the way it does.

**Tests.** Please include them, and test behaviour rather than implementation.
Note that some tests in `x/` do not survive parallel execution and the suite is
run with `-p 1`:

```sh
go test ./x/... -count=1 -p 1
```

**Dependencies.** Added sparingly. If a new one is necessary, say why and what
you tried first.

## Reporting bugs

For anything in `x/cluster/` or `x/mlxrunner/shard/`, open an issue here. For
upstream Ollama behaviour unrelated to split inference, report it to
[ollama/ollama](https://github.com/ollama/ollama/issues), where it can be fixed
for everyone.

Security issues go to [SECURITY.md](SECURITY.md) instead — please do not open a
public issue for them.
