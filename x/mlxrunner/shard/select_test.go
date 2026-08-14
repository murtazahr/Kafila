package shard

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// modelLayers builds a manifest resembling a dense decoder: an embedding, a
// fixed set of tensors per block, a final norm, and an output head. Sizes are
// arbitrary but distinct enough that misattribution shows up in the byte
// counts rather than only in the layer count.
func modelLayers(blocks int) []manifest.ManifestLayer {
	const (
		embedSize = 1000
		blockSize = 100
		normSize  = 10
		headSize  = 1000
	)

	out := []manifest.ManifestLayer{
		{Name: "model.embed_tokens.weight", Size: embedSize, MediaType: "application/vnd.ollama.image.tensor"},
	}
	for i := range blocks {
		for _, suffix := range []string{
			"input_layernorm.weight",
			"self_attn.q_proj.weight",
			"self_attn.k_proj.weight",
			"self_attn.v_proj.weight",
			"self_attn.o_proj.weight",
			"post_attention_layernorm.weight",
			"mlp.gate_proj.weight",
			"mlp.up_proj.weight",
			"mlp.down_proj.weight",
		} {
			out = append(out, manifest.ManifestLayer{
				Name:      fmt.Sprintf("model.layers.%d.%s", i, suffix),
				Size:      blockSize,
				MediaType: "application/vnd.ollama.image.tensor",
			})
		}
	}
	return append(out,
		manifest.ManifestLayer{Name: "model.norm.weight", Size: normSize, MediaType: "application/vnd.ollama.image.tensor"},
		manifest.ManifestLayer{Name: "lm_head.weight", Size: headSize, MediaType: "application/vnd.ollama.image.tensor"},
	)
}

func TestSelectLayersCoversModelExactlyOnce(t *testing.T) {
	const blocks = 32
	layers := modelLayers(blocks)

	as, err := Split(blocks, 3)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}

	// Every block tensor must land on exactly one shard. Non-block tensors are
	// counted separately since they follow role rather than range.
	blockHomes := make(map[string]int)
	for _, a := range as {
		for _, l := range SelectLayers(layers, a.Range, a.Role, Model{Blocks: blocks}).Layers {
			if kind, _ := Classify(l.Name); kind == KindBlock {
				blockHomes[l.Name]++
			}
		}
	}

	wantBlockTensors := 0
	for _, l := range layers {
		if kind, _ := Classify(l.Name); kind == KindBlock {
			wantBlockTensors++
		}
	}

	if len(blockHomes) != wantBlockTensors {
		t.Fatalf("shards cover %d block tensors, model has %d", len(blockHomes), wantBlockTensors)
	}
	for name, n := range blockHomes {
		if n != 1 {
			t.Errorf("block tensor %q assigned to %d shards, want 1", name, n)
		}
	}
}

// Non-block tensors must be present somewhere, or the pipeline is missing an
// embedding, a final norm, or an output projection.
func TestSelectLayersPlacesNonBlockTensors(t *testing.T) {
	const blocks = 32
	layers := modelLayers(blocks)

	as, err := Split(blocks, 3)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}

	sels := make([]Selection, len(as))
	for i, a := range as {
		sels[i] = SelectLayers(layers, a.Range, a.Role, Model{Blocks: blocks})
	}

	has := func(sel Selection, name string) bool {
		for _, l := range sel.Layers {
			if l.Name == name {
				return true
			}
		}
		return false
	}

	// Both ends of the token loop belong to the head.
	if !has(sels[0], "model.embed_tokens.weight") {
		t.Error("head shard is missing the token embedding")
	}
	if !has(sels[0], "lm_head.weight") {
		t.Error("head shard is missing the output projection")
	}
	if has(sels[0], "model.norm.weight") {
		t.Error("head shard should not hold the final norm")
	}

	// The tail owns only the final norm and returns a normed hidden state.
	if !has(sels[2], "model.norm.weight") {
		t.Error("tail shard is missing the final norm")
	}
	for _, name := range []string{"model.embed_tokens.weight", "lm_head.weight"} {
		if has(sels[2], name) {
			t.Errorf("tail shard should not hold %q", name)
		}
	}

	for _, name := range []string{"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} {
		if has(sels[1], name) {
			t.Errorf("middle shard should not hold %q", name)
		}
	}
}

// The embedding table is the largest non-block tensor and, under tied word
// embeddings, doubles as the output projection. Storing it on more than one
// node would give back most of what sharding saves, so exactly one shard may
// hold it.
func TestSelectLayersStoresEmbeddingOnce(t *testing.T) {
	const blocks = 28 // Qwen3-0.6B
	layers := modelLayers(blocks)

	for _, n := range []int{1, 2, 3, 4} {
		as, err := Split(blocks, n)
		if err != nil {
			t.Fatalf("Split(%d, %d) failed: %v", blocks, n, err)
		}

		holders := 0
		for _, a := range as {
			for _, l := range SelectLayers(layers, a.Range, a.Role, Model{Blocks: blocks}).Layers {
				if kind, _ := Classify(l.Name); kind == KindEmbedding {
					holders++
				}
			}
		}
		if holders != 1 {
			t.Errorf("%d shards: embedding held by %d nodes, want exactly 1", n, holders)
		}
	}
}

// The point of sharding is that a node reads less than the whole model. If a
// shard's byte count approaches the total, capacity is not distributing and the
// split buys only network hops.
func TestSelectLayersDistributesBytes(t *testing.T) {
	const blocks = 32
	layers := modelLayers(blocks)

	as, err := Split(blocks, 4)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}

	var total int64
	for _, l := range layers {
		total += l.Size
	}

	for i, a := range as {
		sel := SelectLayers(layers, a.Range, a.Role, Model{Blocks: blocks})

		if sel.TotalBytes() != total {
			t.Errorf("shard %d accounts for %d bytes, manifest has %d", i, sel.TotalBytes(), total)
		}
		if sel.Skipped == 0 {
			t.Errorf("shard %d skipped nothing; it is holding the whole model", i)
		}
		if sel.Fraction() >= 0.5 {
			t.Errorf("shard %d holds %.0f%% of the model across 4 shards", i, 100*sel.Fraction())
		}
	}
}

// A cluster of one must read the entire model, byte for byte. This is the
// Stage 1 case, where the distributed path has to behave exactly like the
// unsharded runner.
func TestSelectLayersSingleShardKeepsEverything(t *testing.T) {
	const blocks = 32
	layers := modelLayers(blocks)

	sel := SelectLayers(layers, Range{Start: 0, End: blocks}, Head|Tail, Model{Blocks: blocks})

	if sel.Skipped != 0 {
		t.Errorf("single shard skipped %d layers, want 0", sel.Skipped)
	}
	if len(sel.Layers) != len(layers) {
		t.Errorf("single shard kept %d layers, want %d", len(sel.Layers), len(layers))
	}
	if sel.Fraction() != 1 {
		t.Errorf("single shard holds %.2f of the model, want 1", sel.Fraction())
	}
}

// A layer with no name cannot be attributed to a block, so it must be kept
// rather than silently dropped.
func TestSelectLayersKeepsUnnamedLayers(t *testing.T) {
	layers := []manifest.ManifestLayer{
		{Name: "", Size: 42},
		{Name: "model.layers.99.mlp.up_proj.weight", Size: 100},
	}

	sel := SelectLayers(layers, Range{Start: 0, End: 4}, 0, Model{Blocks: 4})

	if len(sel.Layers) != 1 || sel.Layers[0].Size != 42 {
		t.Fatalf("expected the unnamed layer to be kept, got %+v", sel.Layers)
	}
}

// Selecting with the tensor map remapper must produce names a model expecting
// blocks numbered from zero can bind. This is the end-to-end contract the
// key-remap approach rests on.
func TestSelectThenRemapYieldsZeroBasedBlocks(t *testing.T) {
	const blocks = 32
	layers := modelLayers(blocks)

	r := Range{Start: 11, End: 22}
	sel := SelectLayers(layers, r, 0, Model{Blocks: blocks})

	tensors := make(map[string]int, len(sel.Layers))
	for _, l := range sel.Layers {
		tensors[l.Name] = 1
	}

	local := RemapAll(tensors, r)

	seen := make(map[int]bool)
	for name := range local {
		idx, ok := BlockIndex(name)
		if !ok {
			continue
		}
		if idx < 0 || idx >= r.Len() {
			t.Errorf("remapped tensor %q has index %d, outside [0,%d)", name, idx, r.Len())
		}
		seen[idx] = true
	}

	if len(seen) != r.Len() {
		t.Errorf("remapped tensors cover %d blocks, want %d", len(seen), r.Len())
	}
}
