package mlxrunner

import (
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// Inspect derives the placement-relevant shape of a model from its manifest,
// without loading any weights.
//
// A planner needs two facts before a single tensor is read: how many
// transformer blocks there are to divide, and whether the output projection is
// tied to the embedding. Both are cheap to obtain — the block count comes from
// tensor names in the manifest and the tie from config.json — so planning can
// happen before anything is committed to a node.
func Inspect(modelName string) (shard.Model, error) {
	root, err := model.Open(modelName)
	if err != nil {
		return shard.Model{}, fmt.Errorf("inspect %s: %w", modelName, err)
	}
	defer root.Close()

	return inspectRoot(root)
}

func inspectRoot(root *model.Root) (shard.Model, error) {
	layers := root.Manifest.GetTensorLayers("")
	if len(layers) == 0 {
		return shard.Model{}, fmt.Errorf("model has no tensor layers; it is not a safetensors model")
	}

	// The block count comes from the tensors themselves rather than from
	// config.json. What can be sharded is what is actually present, and a
	// config claiming more layers than the checkpoint ships would produce a
	// plan with a stage that owns nothing.
	blocks := 0
	for _, l := range layers {
		if kind, idx := shard.Classify(l.Name); kind == shard.KindBlock && idx+1 > blocks {
			blocks = idx + 1
		}
	}
	if blocks == 0 {
		return shard.Model{}, fmt.Errorf("no transformer blocks found among %d tensors", len(layers))
	}

	m := shard.Model{Blocks: blocks}

	// The tie is a property of the architecture, not of the tensor names: a
	// tied checkpoint still ships a separate lm_head holding a copy of the
	// embedding, so it can only be recognised from config.
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		// Without the tie flag the redundant lm_head is loaded, which costs
		// memory on the head but is not wrong. Worth a warning, not a failure.
		slog.Warn("cluster: could not read config.json; assuming untied embeddings", "error", err)
		return m, nil
	}

	var cfg struct {
		TieWordEmbeddings bool `json:"tie_word_embeddings"`
		NumHiddenLayers   int  `json:"num_hidden_layers"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		slog.Warn("cluster: could not parse config.json; assuming untied embeddings", "error", err)
		return m, nil
	}

	m.TiedEmbeddings = cfg.TieWordEmbeddings

	if cfg.NumHiddenLayers > 0 && cfg.NumHiddenLayers != blocks {
		// Not fatal — the tensors win — but it means one of the two is lying
		// about the model, which is worth knowing before trusting a plan.
		slog.Warn("cluster: block count disagrees with config",
			"from_tensors", blocks, "num_hidden_layers", cfg.NumHiddenLayers)
	}

	return m, nil
}
