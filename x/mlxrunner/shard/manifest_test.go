package shard

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// TestAgainstRealManifest exercises selection against a manifest produced by
// `ollama create --experimental`, rather than the synthetic fixture the other
// tests use. Point it at one with:
//
//	OLLAMA_TEST_MANIFEST=~/.ollama/models/manifests/registry.ollama.ai/library/qwen3-mlx/0.6b \
//	    go test ./x/mlxrunner/shard/ -run TestAgainstRealManifest -v
//
// Real manifests are the only place the tensor naming conventions are pinned
// down; everything else in this package is an assumption about them.
func TestAgainstRealManifest(t *testing.T) {
	path := os.Getenv("OLLAMA_TEST_MANIFEST")
	if path == "" {
		t.Skip("set OLLAMA_TEST_MANIFEST to a manifest produced by `ollama create --experimental`")
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	var mf manifest.Manifest
	if err := json.Unmarshal(raw, &mf); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}

	var tensors []manifest.ManifestLayer
	for _, l := range mf.Layers {
		if l.MediaType == "application/vnd.ollama.image.tensor" {
			tensors = append(tensors, l)
		}
	}
	if len(tensors) == 0 {
		t.Fatal("manifest has no tensor layers; is this a GGUF model?")
	}

	// Derive the block count from the names rather than trusting a config.
	blocks := 0
	byKind := map[Kind]int{}
	var totalBytes int64
	for _, l := range tensors {
		kind, idx := Classify(l.Name)
		byKind[kind]++
		totalBytes += l.Size
		if kind == KindBlock && idx+1 > blocks {
			blocks = idx + 1
		}
	}

	t.Logf("%d tensor layers, %.1f MiB, %d blocks", len(tensors), float64(totalBytes)/(1<<20), blocks)
	for _, k := range []Kind{KindEmbedding, KindBlock, KindFinalNorm, KindOutputHead, KindUnknown} {
		t.Logf("  %-12s %d", k, byKind[k])
	}

	if blocks == 0 {
		t.Fatal("no block tensors classified; the naming convention differs from what Classify expects")
	}
	if byKind[KindEmbedding] == 0 {
		t.Error("no embedding tensor classified")
	}
	if byKind[KindFinalNorm] == 0 {
		t.Error("no final norm tensor classified")
	}
	if byKind[KindUnknown] > 0 {
		// Not fatal: unknown tensors are replicated to every shard, which is
		// safe but wasteful. Worth knowing about.
		for _, l := range tensors {
			if k, _ := Classify(l.Name); k == KindUnknown {
				t.Logf("  unclassified (replicated to all shards): %s", l.Name)
			}
		}
	}

	// A tied checkpoint still ships an lm_head duplicating the embedding, so
	// report both readings: the difference is what declaring the tie saves on
	// the head, which is the node carrying the most weight already.
	for _, m := range []Model{{Blocks: blocks}, {Blocks: blocks, TiedEmbeddings: true}} {
		t.Logf("--- TiedEmbeddings=%v ---", m.TiedEmbeddings)
		runManifestSplits(t, tensors, m, totalBytes)
	}
}

func runManifestSplits(t *testing.T, tensors []manifest.ManifestLayer, m Model, totalBytes int64) {
	t.Helper()

	blocks := m.Blocks
	for _, n := range []int{1, 2, 3, 4} {
		if n > blocks {
			continue
		}

		as, err := m.Split(n)
		if err != nil {
			t.Fatalf("Split(%d, %d): %v", blocks, n, err)
		}
		if err := Validate(as, blocks); err != nil {
			t.Fatalf("invalid plan for %d shards: %v", n, err)
		}

		blockHomes := map[string]int{}
		embedHomes := 0
		var maxFraction float64

		for i, a := range as {
			sel := SelectLayers(tensors, a.Range, a.Role, m)

			if sel.TotalBytes() != totalBytes {
				t.Errorf("%d shards, shard %d: accounts for %d bytes, manifest has %d",
					n, i, sel.TotalBytes(), totalBytes)
			}

			for _, l := range sel.Layers {
				switch k, _ := Classify(l.Name); k {
				case KindBlock:
					blockHomes[l.Name]++
				case KindEmbedding:
					embedHomes++
				}
			}

			maxFraction = max(maxFraction, sel.Fraction())
			t.Logf("  %d shards | shard %d %-9s %-8s %6.1f MiB (%4.1f%%)",
				n, i, a.Range, a.Role, float64(sel.KeptBytes)/(1<<20), 100*sel.Fraction())
		}

		for name, count := range blockHomes {
			if count != 1 {
				t.Errorf("%d shards: block tensor %q on %d shards, want 1", n, name, count)
			}
		}
		if embedHomes != 1 {
			t.Errorf("%d shards: embedding on %d shards, want exactly 1", n, embedHomes)
		}

		// With more than one shard, no node should still be holding most of
		// the model, or capacity has not meaningfully distributed.
		if n > 1 && maxFraction > 0.9 {
			t.Errorf("%d shards: largest shard still holds %.0f%% of the model", n, 100*maxFraction)
		}
	}
}
