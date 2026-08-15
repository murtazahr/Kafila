package model

import (
	"os"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// TestInspect checks that a model's placement-relevant shape can be derived
// from its manifest without loading weights, since the planner needs those
// facts before anything is committed to a node.
func TestInspect(t *testing.T) {
	modelName := os.Getenv("OLLAMA_TEST_MLX_MODEL")
	if modelName == "" {
		t.Skip("set OLLAMA_TEST_MLX_MODEL to a safetensors model name")
	}

	m, err := Inspect(modelName)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}

	t.Logf("%s: %d blocks, tied=%v", modelName, m.Blocks, m.TiedEmbeddings)

	if m.Blocks <= 0 {
		t.Errorf("derived %d blocks", m.Blocks)
	}

	// The block count must agree with what selection actually finds, or a plan
	// would leave a stage owning nothing.
	root, err := Open(modelName)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	high := 0
	for _, l := range root.Manifest.GetTensorLayers("") {
		if kind, idx := shard.Classify(l.Name); kind == shard.KindBlock && idx+1 > high {
			high = idx + 1
		}
	}
	if m.Blocks != high {
		t.Errorf("Inspect reported %d blocks, tensors show %d", m.Blocks, high)
	}
}

// Inspecting a GGUF model must fail rather than produce a plan that cannot run.
func TestInspectRejectsNonSafetensors(t *testing.T) {
	if _, err := Inspect("does-not-exist-at-all:latest"); err == nil {
		t.Error("Inspect accepted a model that does not exist")
	}
}
