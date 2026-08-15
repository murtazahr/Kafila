package mlxrunner

import (
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// LoadShardTensors loads only the tensors a shard needs, named in the shard's
// own block numbering.
//
// Selection happens on manifest metadata, so a blob belonging to a block this
// shard does not own is never opened, mapped, or read. Ollama's MLX import
// writes one blob per tensor and records the tensor name on the layer, which is
// what lets the filter run before any file is touched.
//
// The filter is not what keeps device memory proportional to the shard: MLX
// loads lazily, so an unfiltered load that only ever evaluates one shard's
// tensors costs the same resident memory (measured identical, to 0.1 MiB, in
// TestShardLoadMemory). What it saves is file work — opening, mapping and
// header-scanning every blob in the model on every node — and it removes the
// possibility of a shard binding a block it does not own. Note this was
// measured on Metal with unified memory; a discrete-VRAM backend may not defer
// in the same way.
//
// Names come back remapped into the shard's local index space — block
// a.Range.Start becomes block 0 — so an unmodified model implementation can
// bind them. Callers must set the model's block count to a.Range.Len() to
// match.
func LoadShardTensors(root *model.Root, a shard.Assignment, m shard.Model) (map[string]*mlx.Array, shard.Selection, error) {
	sel := shard.SelectLayers(root.Manifest.GetTensorLayers(""), a.Range, a.Role, m)

	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range sel.Layers {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		for name, arr := range mlx.Load(root.Manifest.BlobPath(layer.Digest)) {
			rawTensors[name] = arr
		}
	}

	// Remap after normalizing suffixes: the quantization companions carry the
	// same block index as the weight they belong to, so rewriting indices last
	// keeps a weight and its scale together.
	tensors := shard.RemapAll(normalizeQuantSuffixes(rawTensors), a.Range)

	slog.Info("loaded shard tensors",
		"range", a.Range, "role", a.Role, "count", len(tensors),
		"kept_bytes", sel.KeptBytes, "skipped_bytes", sel.SkippedBytes,
		"skipped_layers", sel.Skipped, "fraction", sel.Fraction())

	return tensors, sel, nil
}
