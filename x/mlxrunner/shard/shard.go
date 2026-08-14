// Package shard splits a model across nodes at transformer-block boundaries.
//
// A shard owns a contiguous range of blocks and, depending on its position in
// the pipeline, the embedding that precedes the first block or the final norm
// and output projection that follow the last one. Middle shards own neither:
// they receive a hidden state, run their blocks, and pass the hidden state on.
//
// The design leans on an existing property of the model implementations in
// x/models: Forward ranges over the Layers slice and NewCaches sizes itself
// from len(Layers), so a model holding fewer blocks than its config declares is
// already coherent. Only LoadWeights bakes in absolute block indices, and
// Remap in this package sidesteps that by rewriting tensor names into the
// shard's local index space before the model ever sees them.
package shard

import "fmt"

// Range is a half-open range of transformer block indices, [Start, End).
type Range struct {
	Start int
	End   int
}

// Len returns the number of blocks in the range.
func (r Range) Len() int { return r.End - r.Start }

// Contains reports whether block i falls in the range.
func (r Range) Contains(i int) bool { return i >= r.Start && i < r.End }

func (r Range) String() string { return fmt.Sprintf("[%d,%d)", r.Start, r.End) }

// Role describes which non-block components a shard owns. A single-node
// deployment holds both roles; a middle shard holds neither.
//
// The pipeline is a ring rather than a line: the head owns both ends of the
// token loop, embedding the input token and unembedding the hidden state that
// comes back round. That placement is what keeps the embedding table on one
// node. With tied word embeddings the output projection is the embedding, so
// splitting the two ends across different nodes would store it twice — on
// Qwen3-0.6B that is 297 MiB duplicated out of a 1.4 GiB model, which is most
// of what sharding was meant to save. The cost is that the tail returns a
// hidden state instead of a token id, growing the backward edge from 4 bytes
// to hidden_size * dtype (2 KiB at hidden 1024 in bf16). Against a forward hop
// of the same size, that is a good trade.
type Role uint8

const (
	// Head owns the token embedding, the output projection, and the sampler.
	// It starts the forward pass from token ids and finishes it by turning the
	// returned hidden state into the next token.
	Head Role = 1 << iota
	// Tail owns the final norm applied after the last block, and returns a
	// normed hidden state rather than a sampled token.
	Tail
)

// IsHead reports whether the shard begins the pipeline.
func (r Role) IsHead() bool { return r&Head != 0 }

// IsTail reports whether the shard ends the pipeline.
func (r Role) IsTail() bool { return r&Tail != 0 }

func (r Role) String() string {
	switch {
	case r.IsHead() && r.IsTail():
		return "head+tail"
	case r.IsHead():
		return "head"
	case r.IsTail():
		return "tail"
	default:
		return "middle"
	}
}

// Assignment is one node's slice of the model.
type Assignment struct {
	Range Range
	Role  Role
}

// Model captures the facts about a checkpoint that change placement decisions.
type Model struct {
	// Blocks is the number of transformer blocks.
	Blocks int

	// TiedEmbeddings reports whether the model's output projection is the
	// token embedding, as declared by tie_word_embeddings in config.json.
	//
	// It matters because a tied checkpoint still ships a separate lm_head
	// tensor holding a byte-for-byte copy of the embedding, and the model
	// implementations ignore it — they rebuild the projection from the
	// embedding instead. Loading it is wasted work on the node that can least
	// afford it: on Qwen3-0.6B that copy is 297 MiB, a fifth of the model,
	// landing on the head that already owns the embedding, the sampler, and
	// its own blocks.
	TiedEmbeddings bool
}

// Split divides this model's blocks across n shards.
func (m Model) Split(n int) ([]Assignment, error) { return Split(m.Blocks, n) }

// Split divides blocks across n shards as evenly as possible, giving the
// earlier shards the extra block when the division is uneven. It is the
// baseline placement policy: it assumes homogeneous nodes and ignores both
// per-layer weight size and device capability. Capability-aware planners
// replace this, and are the point at which placement becomes a research
// variable rather than a fixed rule.
func Split(blocks, n int) ([]Assignment, error) {
	if blocks <= 0 {
		return nil, fmt.Errorf("shard: block count must be positive, got %d", blocks)
	}
	if n <= 0 {
		return nil, fmt.Errorf("shard: shard count must be positive, got %d", n)
	}
	if n > blocks {
		return nil, fmt.Errorf("shard: cannot split %d blocks across %d shards", blocks, n)
	}

	out := make([]Assignment, n)
	base, extra := blocks/n, blocks%n

	start := 0
	for i := range out {
		size := base
		if i < extra {
			size++
		}

		var role Role
		if i == 0 {
			role |= Head
		}
		if i == n-1 {
			role |= Tail
		}

		out[i] = Assignment{Range: Range{Start: start, End: start + size}, Role: role}
		start += size
	}

	return out, nil
}

// Validate reports whether assignments tile [0, blocks) exactly once, in order,
// with the head and tail roles landing on the first and last shards. A plan
// that fails this check would produce silently wrong output rather than an
// error, so callers should treat a failure as fatal.
func Validate(as []Assignment, blocks int) error {
	if len(as) == 0 {
		return fmt.Errorf("shard: empty assignment")
	}

	next := 0
	for i, a := range as {
		if a.Range.Start != next {
			return fmt.Errorf("shard: assignment %d starts at %d, expected %d", i, a.Range.Start, next)
		}
		if a.Range.Len() <= 0 {
			return fmt.Errorf("shard: assignment %d is empty: %s", i, a.Range)
		}
		if got, want := a.Role.IsHead(), i == 0; got != want {
			return fmt.Errorf("shard: assignment %d head role is %v, expected %v", i, got, want)
		}
		if got, want := a.Role.IsTail(), i == len(as)-1; got != want {
			return fmt.Errorf("shard: assignment %d tail role is %v, expected %v", i, got, want)
		}
		next = a.Range.End
	}

	if next != blocks {
		return fmt.Errorf("shard: assignments cover %d blocks, model has %d", next, blocks)
	}

	return nil
}
