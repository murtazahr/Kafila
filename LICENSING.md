# Licensing

Kafila is dual-licensed. The same code is available under two sets of terms,
and you choose the one that fits what you are doing.

| | Noncommercial | Commercial |
|---|---|---|
| **Terms** | [PolyForm Noncommercial 1.0.0](LICENSE) | Negotiated |
| **Cost** | Free | Contact the licensor |
| **Covers** | Research, teaching, study, personal projects, evaluation | Anything the noncommercial terms do not permit |
| **Source** | Provided | Provided |

If you are a university, a student, an independent researcher, or anyone using
Kafila for something other than commercial advantage, the noncommercial terms
already cover you and there is nothing to sign or pay.

If you want to use Kafila in a product, in a service you charge for, or
internally at a company as part of commercial operations, you need a commercial
licence.

## Why not a conventional open-source licence

Kafila is a research platform. Keeping it free for the research community is the
point, and the commercial licence is what makes it possible to keep working on
it. PolyForm Noncommercial states that intent directly rather than approximating
it with a copyleft licence that a well-resourced company can simply comply with.

The trade-off is real and worth stating: **PolyForm Noncommercial is not an
OSI-approved open-source licence.** Some institutions restrict the use of
source-available software, and some venues treat it as less open than a
permissive or copyleft licence. That is a deliberate cost, accepted in exchange
for the commercial option.

## What is Kafila's own work

The commercial licence can only be offered for code whose copyright the licensor
holds. Kafila is a fork of Ollama, and the following are Kafila's own
contributions:

```
x/cluster/                     the ring, the wire format, the trace
                               discipline, the operations console
x/mlxrunner/shard/             model partitioning, tensor classification,
                               key remapping
x/mlxrunner/shardnode.go       the node entrypoint
x/mlxrunner/model/shardload.go, inspect.go
scripts/split_cluster.sh
scripts/build_worldmap.py
research/
```

Changes to existing upstream files — `x/mlxrunner/pipeline.go`,
`x/mlxrunner/prefix_cache.go`, `x/mlxrunner/model/base/base.go`,
`x/mlxrunner/runner.go`, `x/models/qwen3/qwen3.go` and others — are Kafila's, but
sit inside files that are otherwise Ollama's.

Everything else originates upstream and remains available from Ollama under the
MIT licence, whatever terms Kafila is distributed under. See [NOTICE](NOTICE)
and [LICENSE-OLLAMA-MIT](LICENSE-OLLAMA-MIT).

## Contributions

**Dual licensing only works if one party holds the copyright to everything being
dual-licensed.** A contribution accepted without an agreement cannot be included
in a commercial licence, because the licensor has no right to relicense someone
else's code.

Contributions therefore require a Contributor Licence Agreement granting a
licence broad enough to permit commercial sublicensing.

Please open an issue or get in touch before starting work on a pull request, so
the agreement is in place first.

## Obtaining a commercial licence

Contact **Murtaza Rangwala** — <murtazahatimr@icloud.com>.

Kafila's copyright is held by Murtaza Rangwala personally, so commercial terms
can be granted directly.
