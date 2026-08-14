package shard

import "testing"

func TestBlockIndex(t *testing.T) {
	for name, tc := range map[string]struct {
		tensor string
		want   int
		ok     bool
	}{
		"plain":             {"model.layers.11.self_attn.q_proj.weight", 11, true},
		"zero":              {"model.layers.0.input_layernorm.weight", 0, true},
		"multi digit":       {"model.layers.127.mlp.down_proj.weight", 127, true},
		"language prefix":   {"language_model.model.layers.3.mlp.up_proj.weight", 3, true},
		"nested prefix":     {"model.language_model.layers.0.moe.experts", 0, true},
		"quantized scale":   {"model.layers.7.self_attn.k_proj.weight_scale", 7, true},
		"leading marker":    {"layers.5.mlp.gate_proj.weight", 5, true},
		"embedding":         {"model.embed_tokens.weight", 0, false},
		"final norm":        {"model.norm.weight", 0, false},
		"output head":       {"lm_head.weight", 0, false},
		"substring guard":   {"model.sublayers.4.weight", 0, false},
		"no index":          {"model.layers.weight", 0, false},
		"non numeric":       {"model.layers.foo.weight", 0, false},
		"index at end":      {"model.layers.9", 0, false},
		"negative rejected": {"model.layers.-1.weight", 0, false},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := BlockIndex(tc.tensor)
			if ok != tc.ok {
				t.Fatalf("BlockIndex(%q) ok = %v, want %v", tc.tensor, ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("BlockIndex(%q) = %d, want %d", tc.tensor, got, tc.want)
			}
		})
	}
}

// A name whose prefix contains a decoy "layers." that does not carry an index
// must still resolve to the real block index further along.
func TestBlockIndexSkipsUnindexedMarker(t *testing.T) {
	const tensor = "vision.layers.pooled.model.layers.6.mlp.up_proj.weight"
	got, ok := BlockIndex(tensor)
	if !ok {
		t.Fatalf("BlockIndex(%q) failed to find an index", tensor)
	}
	if got != 6 {
		t.Errorf("BlockIndex(%q) = %d, want 6", tensor, got)
	}
}

func TestClassify(t *testing.T) {
	for name, tc := range map[string]struct {
		tensor string
		kind   Kind
		idx    int
	}{
		"embedding":        {"model.embed_tokens.weight", KindEmbedding, -1},
		"tied embedding":   {"language_model.model.embed_tokens.weight", KindEmbedding, -1},
		"block attention":  {"model.layers.11.self_attn.q_proj.weight", KindBlock, 11},
		"block norm":       {"model.layers.11.input_layernorm.weight", KindBlock, 11},
		"block post norm":  {"model.layers.2.post_attention_layernorm.weight", KindBlock, 2},
		"final norm":       {"model.norm.weight", KindFinalNorm, -1},
		"final norm quant": {"model.norm.weight_scale", KindFinalNorm, -1},
		"final layernorm":  {"model.final_layernorm.weight", KindFinalNorm, -1},
		"output head":      {"lm_head.weight", KindOutputHead, -1},
		"prefixed head":    {"language_model.lm_head.weight", KindOutputHead, -1},
		"rope freqs":       {"rope.freqs", KindUnknown, -1},
	} {
		t.Run(name, func(t *testing.T) {
			kind, idx := Classify(tc.tensor)
			if kind != tc.kind {
				t.Errorf("Classify(%q) kind = %v, want %v", tc.tensor, kind, tc.kind)
			}
			if idx != tc.idx {
				t.Errorf("Classify(%q) idx = %d, want %d", tc.tensor, idx, tc.idx)
			}
		})
	}
}

// A block's own layernorms must never be mistaken for the model's final norm,
// which would hand every shard a tensor only the tail should hold.
func TestClassifyBlockNormIsNotFinalNorm(t *testing.T) {
	for _, tensor := range []string{
		"model.layers.0.input_layernorm.weight",
		"model.layers.31.post_attention_layernorm.weight",
		"model.layers.4.self_attn.q_norm.weight",
	} {
		if kind, _ := Classify(tensor); kind != KindBlock {
			t.Errorf("Classify(%q) = %v, want block", tensor, kind)
		}
	}
}

func TestKeep(t *testing.T) {
	middle := Range{Start: 11, End: 22}

	tied := Model{Blocks: 32, TiedEmbeddings: true}

	for name, tc := range map[string]struct {
		tensor string
		rng    Range
		role   Role
		model  Model
		want   bool
	}{
		// A tied checkpoint ships an lm_head that duplicates the embedding and
		// that no model implementation reads, so it is skipped everywhere.
		"tied drops output head on head": {"lm_head.weight", Range{0, 11}, Head, tied, false},
		"tied keeps embedding on head":   {"model.embed_tokens.weight", Range{0, 11}, Head, tied, true},
		"tied drops output head on tail": {"lm_head.weight", Range{22, 32}, Tail, tied, false},
		"tied single node drops head":    {"lm_head.weight", Range{0, 32}, Head | Tail, tied, false},
		"tied still keeps blocks":        {"model.layers.5.mlp.up_proj.weight", Range{0, 11}, Head, tied, true},
		"middle keeps own block":         {"model.layers.15.mlp.up_proj.weight", middle, 0, Model{}, true},
		"middle drops earlier block":     {"model.layers.3.mlp.up_proj.weight", middle, 0, Model{}, false},
		"middle drops later block":       {"model.layers.28.mlp.up_proj.weight", middle, 0, Model{}, false},
		"middle drops embedding":         {"model.embed_tokens.weight", middle, 0, Model{}, false},
		"middle drops final norm":        {"model.norm.weight", middle, 0, Model{}, false},
		"middle drops output head":       {"lm_head.weight", middle, 0, Model{}, false},
		"middle keeps unknown":           {"rope.freqs", middle, 0, Model{}, true},
		"head keeps embedding":           {"model.embed_tokens.weight", Range{0, 11}, Head, Model{}, true},
		"head keeps output head":         {"lm_head.weight", Range{0, 11}, Head, Model{}, true},
		"head drops final norm":          {"model.norm.weight", Range{0, 11}, Head, Model{}, false},
		"tail keeps final norm":          {"model.norm.weight", Range{22, 32}, Tail, Model{}, true},
		"tail drops output head":         {"lm_head.weight", Range{22, 32}, Tail, Model{}, false},
		"tail drops embedding":           {"model.embed_tokens.weight", Range{22, 32}, Tail, Model{}, false},
		"single node keeps all":          {"model.norm.weight", Range{0, 32}, Head | Tail, Model{}, true},
		"single node keeps embed":        {"model.embed_tokens.weight", Range{0, 32}, Head | Tail, Model{}, true},
	} {
		t.Run(name, func(t *testing.T) {
			if got := Keep(tc.tensor, tc.rng, tc.role, tc.model); got != tc.want {
				t.Errorf("Keep(%q, %s, %s) = %v, want %v", tc.tensor, tc.rng, tc.role, got, tc.want)
			}
		})
	}
}

func TestRemap(t *testing.T) {
	r := Range{Start: 11, End: 22}

	for name, tc := range map[string]struct {
		tensor string
		want   string
		ok     bool
	}{
		"first block becomes zero": {"model.layers.11.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight", true},
		"interior block":           {"model.layers.15.mlp.up_proj.weight", "model.layers.4.mlp.up_proj.weight", true},
		"last block":               {"model.layers.21.input_layernorm.weight", "model.layers.10.input_layernorm.weight", true},
		"multi to single digit":    {"model.layers.20.mlp.down_proj.weight", "model.layers.9.mlp.down_proj.weight", true},
		"quantized suffix kept":    {"model.layers.12.self_attn.k_proj.weight_scale", "model.layers.1.self_attn.k_proj.weight_scale", true},
		"prefixed name":            {"language_model.model.layers.11.mlp.gate_proj.weight", "language_model.model.layers.0.mlp.gate_proj.weight", true},
		"non block passes through": {"model.embed_tokens.weight", "model.embed_tokens.weight", true},
		"below range dropped":      {"model.layers.10.mlp.up_proj.weight", "", false},
		"above range dropped":      {"model.layers.22.mlp.up_proj.weight", "", false},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := Remap(tc.tensor, r)
			if ok != tc.ok {
				t.Fatalf("Remap(%q) ok = %v, want %v", tc.tensor, ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("Remap(%q) = %q, want %q", tc.tensor, got, tc.want)
			}
		})
	}
}

// The head shard already starts at zero, so remapping must be a no-op for it
// rather than shifting names.
func TestRemapHeadIsIdentity(t *testing.T) {
	r := Range{Start: 0, End: 11}
	const tensor = "model.layers.7.mlp.up_proj.weight"

	got, ok := Remap(tensor, r)
	if !ok {
		t.Fatalf("Remap(%q) dropped a block in range", tensor)
	}
	if got != tensor {
		t.Errorf("Remap(%q) = %q, want unchanged", tensor, got)
	}
}

func TestRemapAll(t *testing.T) {
	in := map[string]int{
		"model.embed_tokens.weight":               1,
		"model.layers.10.mlp.up_proj.weight":      2,
		"model.layers.11.mlp.up_proj.weight":      3,
		"model.layers.12.self_attn.o_proj.weight": 4,
		"model.norm.weight":                       5,
	}

	got := RemapAll(in, Range{Start: 11, End: 13})

	want := map[string]int{
		"model.embed_tokens.weight":              1,
		"model.layers.0.mlp.up_proj.weight":      3,
		"model.layers.1.self_attn.o_proj.weight": 4,
		"model.norm.weight":                      5,
	}

	if len(got) != len(want) {
		t.Fatalf("RemapAll returned %d tensors, want %d: %v", len(got), len(want), got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("RemapAll[%q] = %v, want %v", k, got[k], v)
		}
	}

	if len(in) != 5 {
		t.Errorf("RemapAll modified its input: %v", in)
	}
}

// Remapping the full range of a single-node deployment must leave every name
// untouched, so a cluster of one is byte-identical to no sharding at all.
func TestRemapAllSingleNodeIsIdentity(t *testing.T) {
	in := map[string]int{
		"model.embed_tokens.weight":          1,
		"model.layers.0.mlp.up_proj.weight":  2,
		"model.layers.31.mlp.up_proj.weight": 3,
		"model.norm.weight":                  4,
		"lm_head.weight":                     5,
	}

	got := RemapAll(in, Range{Start: 0, End: 32})

	if len(got) != len(in) {
		t.Fatalf("RemapAll dropped tensors: got %d, want %d", len(got), len(in))
	}
	for k, v := range in {
		if got[k] != v {
			t.Errorf("RemapAll[%q] = %v, want %v", k, got[k], v)
		}
	}
}
