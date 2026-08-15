package agent

import (
	"bytes"
	"context"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// TestPipelineOverTCP runs the split across real sockets rather than in-process
// calls, and checks the result still matches the whole model.
//
// The earlier test proved the arithmetic survives the cut. This one proves it
// survives the transport too: serialization to bytes, a framed round trip, and
// reconstruction on the far side, at every hop. Those are separate ways to be
// wrong — a dtype misread or a shape mismatch would corrupt values without
// touching the block arithmetic at all.
//
//	OLLAMA_TEST_MLX_MODEL=qwen3-mlx:0.6b go test ./x/cluster/agent/ -run TestPipelineOverTCP -v
func TestPipelineOverTCP(t *testing.T) {
	// Two shards evaluating concurrently in one process deadlock inside MLX:
	// one worker blocks in mlx.Eval while another waits for its thread. MLX
	// keeps global state that is not built for several independent models
	// stepping on it at once, and no arrangement of threads on this side fixes
	// that.
	//
	// It is also not the shape the cluster runs in. Every stage is its own
	// process there, one shard and one MLX context each, which is what
	// cmd/shardnode exercises. This is kept because the setup documents the
	// protocol end to end, and because it will pass the moment MLX tolerates
	// it.
	t.Skip("shards must be separate processes; see cmd/shardnode for the end-to-end test")

	modelName := os.Getenv("OLLAMA_TEST_MLX_MODEL")
	if modelName == "" {
		t.Skip("set OLLAMA_TEST_MLX_MODEL to a safetensors model name")
	}
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX runtime unavailable: %v", err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	spec, err := mlxrunner.Inspect(modelName)
	if err != nil {
		t.Fatalf("inspect: %v", err)
	}

	tokens := []int32{9707, 1879, 11, 1246, 525, 498, 3351, 30}
	want := runWhole(t, modelName, spec, tokens)

	plan, err := spec.Split(3)
	if err != nil {
		t.Fatal(err)
	}

	// Load the head locally and serve the other two over TCP. Every stage
	// after the head goes through the socket path, so nothing here exercises a
	// shortcut the cluster would not use.
	// The head takes no thread of its own: this test drives it directly, and
	// it has already locked its OS thread. Giving it one would mean hopping
	// onto a worker from inside that worker and deadlocking.
	head, err := Load(nil, Config{ModelName: modelName, Model: spec, Assignment: plan[0]})
	if err != nil {
		t.Fatalf("load head: %v", err)
	}

	var stages []*Stage
	for i, a := range plan[1:] {
		// Each served shard gets its own pinned MLX thread. A server accepts
		// on arbitrary goroutines, and an array cannot be evaluated from a
		// thread other than the one that created it.
		th, err := StartThread("shard-" + a.Range.String())
		if err != nil {
			t.Fatalf("start mlx thread for stage %d: %v", i+1, err)
		}
		defer th.Stop(context.Background(), nil)

		s, err := Load(th, Config{ModelName: modelName, Model: spec, Assignment: a})
		if err != nil {
			t.Fatalf("load stage %d: %v", i+1, err)
		}

		srv, err := Listen(s, "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen stage %d: %v", i+1, err)
		}
		defer srv.Close()
		go func() { _ = srv.Serve() }()

		client, err := Dial(a.Range.String(), srv.Addr(), a.Range.String(), 5*time.Second)
		if err != nil {
			t.Fatalf("dial stage %d: %v", i+1, err)
		}
		defer client.Close()

		t.Logf("stage %d: blocks %s role %-6s serving on %s", i+1, a.Range, a.Role, srv.Addr())
		stages = append(stages, client)
	}

	p, err := NewPipelineModel(head, stages)
	if err != nil {
		t.Fatalf("compose pipeline: %v", err)
	}

	var traceOut bytes.Buffer
	rec := trace.NewRecorder(&traceOut, "req-tcp", "head", plan[0].Range)
	p.Trace(rec)

	// Everything must start from the same offset, or positions have already
	// drifted before a single token is processed.
	if err := p.Align(); err != nil {
		t.Fatalf("align: %v", err)
	}

	hidden, _ := p.Forward(headBatch(tokens), nil)
	if hidden == nil {
		t.Fatal("pipeline produced no hidden state; see the logged stage error")
	}
	got := floats(hidden)
	if len(got) != len(want) {
		t.Fatalf("pipeline produced %d values, whole model produced %d", len(got), len(want))
	}

	var maxAbs, scale float64
	for i := range want {
		maxAbs = math.Max(maxAbs, math.Abs(float64(want[i]-got[i])))
		scale = math.Max(scale, math.Abs(float64(want[i])))
	}
	t.Logf("over TCP: %d values, peak %.4f, max difference %.6f", len(got), scale, maxAbs)

	if scale == 0 {
		t.Fatal("whole-model output is all zeros; the comparison is meaningless")
	}
	if rel := maxAbs / scale; rel > 0.02 {
		t.Errorf("transport corrupted the result: max relative difference %.4f", rel)
	}

	// Every stage must have absorbed the same number of tokens.
	for _, s := range stages {
		off, err := s.Align(false)
		if err != nil {
			t.Fatalf("align %s: %v", s.Name, err)
		}
		if off != len(tokens) {
			t.Errorf("stage %s rests at offset %d, want %d", s.Name, off, len(tokens))
		}
	}

	// The hops should be in the trace, with each stage's own reported time
	// subtracted out.
	sums, err := trace.Parse(bytes.NewReader(traceOut.Bytes()))
	if err != nil {
		t.Fatalf("parse trace: %v", err)
	}
	sum := sums["req-tcp"]
	if sum == nil {
		t.Fatal("no trace recorded for the request")
	}
	if sum.InconsistentHops > 0 {
		t.Errorf("%d hops report more remote time than the sender observed", sum.InconsistentHops)
	}

	forwards := 0
	for _, h := range sum.Hops {
		if h.Phase != trace.PhaseAlign {
			forwards++
			one, ok := h.OneWay()
			if !ok {
				t.Errorf("hop to %s produced no one-way estimate", h.To)
			}
			t.Logf("hop -> %-9s %7d bytes  rtt %8s  remote %8s  in flight %8s  one way %8s",
				h.To, h.Bytes, h.RoundTrip, h.RemoteDuration, h.InFlight(), one)
		}
	}
	if forwards != len(stages) {
		t.Errorf("recorded %d forward hops, want %d", forwards, len(stages))
	}
}

// A stage must refuse a payload that disagrees with its header rather than
// reinterpreting the bytes into a well-formed array of wrong values.
func TestDecodeBatchRejectsMismatch(t *testing.T) {
	for name, tc := range map[string]struct {
		req     wire.Request
		payload []byte
	}{
		"payload too short": {
			wire.Request{DType: "BF16", Shape: []int{1, 4, 1024}, SeqOffsets: []int32{0}},
			make([]byte, 100),
		},
		"payload too long": {
			wire.Request{DType: "BF16", Shape: []int{1, 1, 8}, SeqOffsets: []int32{0}},
			make([]byte, 4096),
		},
		"unknown dtype": {
			wire.Request{DType: "NOPE", Shape: []int{1, 1, 8}, SeqOffsets: []int32{0}},
			make([]byte, 16),
		},
		"no offsets": {
			wire.Request{DType: "BF16", Shape: []int{1, 1, 8}},
			make([]byte, 16),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := describe(tc.req, tc.payload); err == nil {
				t.Error("describe accepted a frame whose header and payload disagree")
			}
		})
	}
}

func TestClassifyPhase(t *testing.T) {
	prefill := headBatch([]int32{1, 2, 3, 4})
	if phase, _, chunk := classify(prefill); phase != wire.PhasePrefill || chunk != 0 {
		t.Errorf("multi-token batch classified as %s chunk %d", phase, chunk)
	}

	decode := headBatch([]int32{7})
	decode.SeqOffsets = []int32{12}
	if phase, token, _ := classify(decode); phase != wire.PhaseDecode || token != 12 {
		t.Errorf("single-token batch classified as %s token %d", phase, token)
	}
}

func TestNewPipelineModelRequiresHead(t *testing.T) {
	notHead := &Shard{Assignment: shard.Assignment{
		Range: shard.Range{Start: 10, End: 19},
	}}
	if _, err := NewPipelineModel(notHead, nil); err == nil {
		t.Error("composed a pipeline whose local shard is not the head")
	}
	if _, err := NewPipelineModel(nil, nil); err == nil {
		t.Error("composed a pipeline with no local shard")
	}
}
