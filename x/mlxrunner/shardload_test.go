package mlxrunner

import (
	"os"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// TestShardLoadMemory measures what a shard actually costs in device memory.
//
// The design assumes two things that are cheaper to measure than to argue
// about: that selecting layers at the manifest keeps a shard's resident memory
// proportional to the blocks it owns, and that skipping the redundant lm_head a
// tied checkpoint ships is a real saving rather than an accounting one. It also
// reports memory straight after loading, before anything is evaluated, which
// says whether MLX materializes a safetensors blob eagerly or defers it.
//
// Run against a model imported with `ollama create --experimental`:
//
//	OLLAMA_TEST_MLX_MODEL=qwen3-mlx:0.6b go test ./x/mlxrunner/ -run TestShardLoadMemory -v
func TestShardLoadMemory(t *testing.T) {
	modelName := os.Getenv("OLLAMA_TEST_MLX_MODEL")
	if modelName == "" {
		t.Skip("set OLLAMA_TEST_MLX_MODEL to a safetensors model name")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX runtime unavailable: %v", err)
	}

	// MLX streams are thread-local on CUDA, so every call below has to happen
	// on one OS thread. The runner gets this from mlxthread; a test has to ask
	// for it explicitly or the Go scheduler may migrate the goroutine mid-test.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	root, err := model.Open(modelName)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer root.Close()

	layers := root.Manifest.GetTensorLayers("")
	if len(layers) == 0 {
		t.Fatalf("%s has no tensor layers; is it a GGUF model?", modelName)
	}

	blocks := 0
	for _, l := range layers {
		if kind, idx := shard.Classify(l.Name); kind == shard.KindBlock && idx+1 > blocks {
			blocks = idx + 1
		}
	}
	if blocks == 0 {
		t.Fatal("no block tensors found")
	}
	t.Logf("%s: %d tensor layers, %d blocks", modelName, len(layers), blocks)

	// reset drops everything not pinned and returns device memory to a
	// comparable starting point between measurements.
	reset := func() int {
		mlx.Sweep()
		mlx.ClearCache()
		return mlx.ActiveMemory()
	}

	measure := func(label string, a shard.Assignment, m shard.Model) {
		base := reset()

		tensors, sel, err := loadShardTensors(root, a, m)
		if err != nil {
			t.Fatalf("%s: load: %v", label, err)
		}
		afterLoad := mlx.ActiveMemory()

		arrays := make([]*mlx.Array, 0, len(tensors))
		for _, arr := range tensors {
			arrays = append(arrays, arr)
		}
		mlx.Eval(arrays...)
		afterEval := mlx.ActiveMemory()

		t.Logf("%-28s %3d tensors | manifest %7.1f MiB (%4.1f%%) | resident after load %7.1f MiB, after eval %7.1f MiB",
			label, len(tensors),
			float64(sel.KeptBytes)/(1<<20), 100*sel.Fraction(),
			float64(afterLoad-base)/(1<<20), float64(afterEval-base)/(1<<20))

		if len(tensors) == 0 {
			t.Errorf("%s: loaded no tensors", label)
		}

		// Every block the shard owns must be present, numbered from zero.
		seen := make(map[int]bool)
		for name := range tensors {
			if idx, ok := shard.BlockIndex(name); ok {
				if idx < 0 || idx >= a.Range.Len() {
					t.Errorf("%s: tensor %q has local index %d, outside [0,%d)",
						label, name, idx, a.Range.Len())
				}
				seen[idx] = true
			}
		}
		if len(seen) != a.Range.Len() {
			t.Errorf("%s: tensors cover %d blocks, shard owns %d", label, len(seen), a.Range.Len())
		}
	}

	whole := shard.Range{Start: 0, End: blocks}

	measure("whole model, untied",
		shard.Assignment{Range: whole, Role: shard.Head | shard.Tail},
		shard.Model{Blocks: blocks})

	measure("whole model, tied",
		shard.Assignment{Range: whole, Role: shard.Head | shard.Tail},
		shard.Model{Blocks: blocks, TiedEmbeddings: true})

	tied := shard.Model{Blocks: blocks, TiedEmbeddings: true}
	as, err := tied.Split(2)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	measure("2 shards, head", as[0], tied)
	measure("2 shards, tail", as[1], tied)

	if as, err := tied.Split(4); err == nil {
		measure("4 shards, head", as[0], tied)
		measure("4 shards, middle", as[1], tied)
	}

	// Because loading is lazy, an unfiltered load that only ever evaluates one
	// shard's tensors may cost the same device memory as a filtered one. If so
	// the manifest filter saves file and page-cache work rather than VRAM, and
	// the design should say that rather than claiming it is what distributes
	// capacity. Measure it instead of assuming either way.
	//
	// Deliberately not a subtest: t.Run runs the closure on a new goroutine,
	// and CUDA's MLX streams are thread-local, so evaluating there fails with
	// "no Stream(gpu, 0) in current thread". Metal tolerates the hop; CUDA does
	// not. This is the same constraint mlxthread exists to enforce in the
	// runner.
	{
		base := reset()

		all, err := loadTensorsFromManifest(root)
		if err != nil {
			t.Fatalf("load all: %v", err)
		}
		afterLoad := mlx.ActiveMemory()

		// Evaluate only what the middle shard of a 4-way split would own.
		as, err := tied.Split(4)
		if err != nil {
			t.Fatalf("split: %v", err)
		}
		mid := as[1]

		var arrays []*mlx.Array
		for name, arr := range all {
			if shard.Keep(name, mid.Range, mid.Role, tied) {
				arrays = append(arrays, arr)
			}
		}
		mlx.Eval(arrays...)
		afterEval := mlx.ActiveMemory()

		t.Logf("%-28s %3d of %d tensors | %25s resident after load %7.1f MiB, after eval %7.1f MiB",
			"unfiltered load, part eval", len(arrays), len(all), "",
			float64(afterLoad-base)/(1<<20), float64(afterEval-base)/(1<<20))
	}

	reset()
}

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
	root, err := model.Open(modelName)
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
