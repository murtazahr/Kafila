package agent

import (
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// TestSplitMatchesWholeModel is the experiment the whole project rests on: a
// model cut into pieces and run in sequence must produce the same hidden state
// as the same model run intact.
//
// Everything else — transport, planning, tracing — is machinery around this. If
// the arithmetic does not survive the cut then none of it matters, and if it
// does then the remaining work is engineering rather than doubt.
//
//	OLLAMA_TEST_MLX_MODEL=qwen3-mlx:0.6b go test ./x/cluster/agent/ -run TestSplitMatchesWholeModel -v
func TestSplitMatchesWholeModel(t *testing.T) {
	modelName := os.Getenv("OLLAMA_TEST_MLX_MODEL")
	if modelName == "" {
		t.Skip("set OLLAMA_TEST_MLX_MODEL to a safetensors model name")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX runtime unavailable: %v", err)
	}

	// MLX streams are thread-local on CUDA; keep every call on one thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	spec, err := mlxrunner.Inspect(modelName)
	if err != nil {
		t.Fatalf("inspect: %v", err)
	}
	t.Logf("%s: %d blocks, tied=%v", modelName, spec.Blocks, spec.TiedEmbeddings)

	tokens := []int32{9707, 1879, 11, 1246, 525, 498, 3351, 30}

	whole := runWhole(t, modelName, spec, tokens)
	split := runSplit(t, modelName, spec, tokens, 3)

	if len(whole) != len(split) {
		t.Fatalf("whole model produced %d values, split produced %d", len(whole), len(split))
	}

	// bf16 carries ~8 bits of mantissa, and the two paths accumulate through
	// the same blocks in the same order, so they should agree closely. A real
	// mismatch — a dropped block, a wrong cache, a lost position — moves values
	// far more than rounding does.
	var maxAbs, sumAbs float64
	for i := range whole {
		d := math.Abs(float64(whole[i] - split[i]))
		sumAbs += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	meanAbs := sumAbs / float64(len(whole))

	var scale float64
	for _, v := range whole {
		scale = math.Max(scale, math.Abs(float64(v)))
	}

	t.Logf("hidden state: %d values, peak magnitude %.4f", len(whole), scale)
	t.Logf("difference:   max %.6f, mean %.6f", maxAbs, meanAbs)

	if scale == 0 {
		t.Fatal("whole-model hidden state is all zeros; the comparison is meaningless")
	}
	if rel := maxAbs / scale; rel > 0.02 {
		t.Errorf("split output diverges: max relative difference %.4f", rel)
	}
}

// runWhole evaluates the model as a single unsplit shard.
func runWhole(t *testing.T, modelName string, spec shard.Model, tokens []int32) []float32 {
	t.Helper()

	s, err := Load(nil, Config{
		ModelName: modelName,
		Model:     spec,
		Assignment: shard.Assignment{
			Range: shard.Range{Start: 0, End: spec.Blocks},
			Role:  shard.Head | shard.Tail,
		},
	})
	if err != nil {
		t.Fatalf("load whole model: %v", err)
	}

	out, err := s.Forward(headBatch(tokens))
	if err != nil {
		t.Fatalf("whole model forward: %v", err)
	}
	return floats(out)
}

// runSplit evaluates the model as n shards chained in pipeline order, passing
// each stage's hidden state to the next exactly as the transport will.
func runSplit(t *testing.T, modelName string, spec shard.Model, tokens []int32, n int) []float32 {
	t.Helper()

	plan, err := spec.Split(n)
	if err != nil {
		t.Fatalf("split: %v", err)
	}

	shards := make([]*Shard, len(plan))
	for i, a := range plan {
		s, err := Load(nil, Config{ModelName: modelName, Model: spec, Assignment: a})
		if err != nil {
			t.Fatalf("load shard %d %s: %v", i, a.Range, err)
		}
		shards[i] = s
		t.Logf("shard %d: blocks %s role %s", i, a.Range, a.Role)
	}

	b := headBatch(tokens)

	var hidden *mlx.Array
	for i, s := range shards {
		out, err := s.Forward(b)
		if err != nil {
			t.Fatalf("shard %d forward: %v", i, err)
		}
		hidden = out

		// What the next stage receives: the hidden state, the same positions,
		// and no token ids. This is the batch the wire format carries.
		b = &batch.Batch{
			Hidden:       hidden,
			SeqOffsets:   b.SeqOffsets,
			SeqQueryLens: b.SeqQueryLens,
		}
	}

	// Every stage must agree on how many tokens its caches hold, or their
	// positions have drifted apart.
	for i, s := range shards {
		if got, want := s.Offset(), len(tokens); got != want {
			t.Errorf("shard %d rests at offset %d, want %d", i, got, want)
		}
	}

	return floats(hidden)
}

func headBatch(tokens []int32) *batch.Batch {
	return &batch.Batch{
		InputIDs:     mlx.FromValues(tokens, 1, len(tokens)),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(len(tokens))},
	}
}

// floats materializes a hidden state as float32 for comparison. The hidden
// state is bf16, and Floats refuses anything but F32, so the conversion has to
// be explicit and evaluated before it is read.
func floats(a *mlx.Array) []float32 {
	f := a.AsType(mlx.DTypeFloat32)
	mlx.Eval(f)
	return f.Floats()
}
