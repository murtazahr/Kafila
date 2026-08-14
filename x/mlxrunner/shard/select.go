package shard

import (
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Selection is the set of manifest layers a shard will load, together with what
// it skipped.
//
// The byte counts are the evidence that capacity actually distributes: a shard
// that skips nothing is holding the whole model and the split has bought
// nothing but network hops. Reporting them is cheap because the manifest
// records each blob's size, so no file is opened to compute them.
type Selection struct {
	Layers       []manifest.ManifestLayer
	KeptBytes    int64
	SkippedBytes int64
	Skipped      int
}

// TotalBytes returns the size of the full model as described by the manifest.
func (s Selection) TotalBytes() int64 { return s.KeptBytes + s.SkippedBytes }

// Fraction returns the share of model bytes this shard holds, in [0,1].
func (s Selection) Fraction() float64 {
	total := s.TotalBytes()
	if total == 0 {
		return 0
	}
	return float64(s.KeptBytes) / float64(total)
}

// SelectLayers picks the manifest layers a shard needs for its range and role.
//
// Ollama's MLX import writes one blob per tensor and records the tensor's name
// on the manifest layer, so selection happens entirely on manifest metadata:
// a shard never opens, maps, or reads a blob belonging to a block it does not
// own. Layers whose name is empty are always kept, since a nameless layer
// cannot be attributed to a block and dropping it would fail silently.
func SelectLayers(layers []manifest.ManifestLayer, r Range, role Role, m Model) Selection {
	sel := Selection{Layers: make([]manifest.ManifestLayer, 0, len(layers))}

	for _, l := range layers {
		if l.Name != "" && !Keep(l.Name, r, role, m) {
			sel.SkippedBytes += l.Size
			sel.Skipped++
			continue
		}
		sel.Layers = append(sel.Layers, l)
		sel.KeptBytes += l.Size
	}

	return sel
}
