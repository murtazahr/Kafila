package shard

import (
	"strconv"
	"strings"
)

// Kind classifies a tensor by the role it plays in the model, which decides
// which shards need it.
type Kind int

const (
	// KindUnknown covers anything not recognised. Unknown tensors are loaded on
	// every shard: they are typically small (rope frequency tables and the
	// like), and dropping one silently produces wrong output rather than an
	// error. Keeping them is the conservative default.
	KindUnknown Kind = iota
	// KindEmbedding is the token embedding table, needed only by the head.
	KindEmbedding
	// KindBlock belongs to a numbered transformer block.
	KindBlock
	// KindFinalNorm is the norm applied after the last block, needed only by
	// the tail.
	KindFinalNorm
	// KindOutputHead is the output projection, needed only by the tail. Models
	// with tied embeddings have no such tensor and reuse the embedding instead.
	KindOutputHead
)

func (k Kind) String() string {
	switch k {
	case KindEmbedding:
		return "embedding"
	case KindBlock:
		return "block"
	case KindFinalNorm:
		return "final_norm"
	case KindOutputHead:
		return "output_head"
	default:
		return "unknown"
	}
}

const layersMarker = "layers."

// blockSpan locates the block index inside a tensor name, returning the parsed
// index and the byte offsets of its digits.
//
// The "layers.<N>." segment is matched wherever it appears rather than at a
// fixed offset, so names carrying a "language_model." or "model.language_model."
// prefix parse identically to bare ones. Requiring the marker to start the name
// or follow a dot keeps it from matching inside a longer word.
func blockSpan(name string) (idx, lo, hi int, ok bool) {
	for off := 0; ; {
		i := strings.Index(name[off:], layersMarker)
		if i < 0 {
			return 0, 0, 0, false
		}
		i += off

		if i > 0 && name[i-1] != '.' {
			off = i + len(layersMarker)
			continue
		}

		lo = i + len(layersMarker)
		rest := name[lo:]

		end := strings.IndexByte(rest, '.')
		if end <= 0 {
			off = lo
			continue
		}

		n, err := strconv.Atoi(rest[:end])
		if err != nil || n < 0 {
			off = lo
			continue
		}

		return n, lo, lo + end, true
	}
}

// BlockIndex returns the transformer block index encoded in a tensor name.
func BlockIndex(name string) (int, bool) {
	idx, _, _, ok := blockSpan(name)
	return idx, ok
}

// quantSuffixes are appended by the runner when it folds a tensor's companion
// ".scale" and ".bias" entries into the base name. They are stripped before
// classification so a quantized weight classifies the same as a plain one.
var quantSuffixes = []string{"_scale", "_qbias"}

// Classify identifies a tensor's role. The returned index is meaningful only
// for KindBlock, and is -1 otherwise.
func Classify(name string) (Kind, int) {
	if idx, ok := BlockIndex(name); ok {
		return KindBlock, idx
	}

	switch {
	case strings.Contains(name, "embed_tokens"):
		return KindEmbedding, -1
	case strings.Contains(name, "lm_head"):
		return KindOutputHead, -1
	case isFinalNorm(name):
		return KindFinalNorm, -1
	}

	return KindUnknown, -1
}

// isFinalNorm reports whether a non-block tensor is the model's trailing norm.
// Callers must rule out block tensors first: a block's "input_layernorm" would
// otherwise reach here and be misread.
func isFinalNorm(name string) bool {
	base := name
	for _, s := range quantSuffixes {
		base = strings.TrimSuffix(base, s)
	}
	base = strings.TrimSuffix(base, ".weight")
	base = strings.TrimSuffix(base, ".bias")

	if i := strings.LastIndexByte(base, '.'); i >= 0 {
		base = base[i+1:]
	}

	switch base {
	case "norm", "final_norm", "final_layernorm":
		return true
	}
	return false
}

// Keep reports whether a shard with this range and role needs the tensor.
//
// Applied to manifest layer names, this decides membership before any blob is
// opened, so a shard never reads bytes for blocks it does not own.
func Keep(name string, r Range, role Role, m Model) bool {
	kind, idx := Classify(name)
	switch kind {
	case KindBlock:
		return r.Contains(idx)
	case KindEmbedding:
		// Both ends of the token loop live on the head, so the embedding and
		// the output projection land on the same node.
		return role.IsHead()
	case KindOutputHead:
		// Redundant under tied embeddings: it duplicates the embedding the
		// head already holds, and no model implementation reads it.
		return role.IsHead() && !m.TiedEmbeddings
	case KindFinalNorm:
		return role.IsTail()
	default:
		return true
	}
}

// Remap rewrites a tensor name from the model's absolute block numbering into
// the shard's local numbering, so that block r.Start becomes block 0.
//
// This is what lets an unmodified model implementation load a shard. Every
// model in x/models addresses its weights purely by string name and sizes its
// layer slice from config, so a model told it has r.Len() blocks, handed
// tensors renumbered from zero, builds a correct shard with no changes to any
// model file.
//
// Names without a block index pass through unchanged. The second return value
// is false when the tensor belongs to a block this shard does not own, in which
// case the caller should drop it.
func Remap(name string, r Range) (string, bool) {
	idx, lo, hi, ok := blockSpan(name)
	if !ok {
		return name, true
	}
	if !r.Contains(idx) {
		return "", false
	}

	var b strings.Builder
	b.Grow(len(name))
	b.WriteString(name[:lo])
	b.WriteString(strconv.Itoa(idx - r.Start))
	b.WriteString(name[hi:])
	return b.String(), true
}

// RemapAll applies Remap across a tensor map, dropping entries outside the
// range. The input map is not modified.
func RemapAll[T any](tensors map[string]T, r Range) map[string]T {
	out := make(map[string]T, len(tensors))
	for name, v := range tensors {
		local, ok := Remap(name, r)
		if !ok {
			continue
		}
		out[local] = v
	}
	return out
}
