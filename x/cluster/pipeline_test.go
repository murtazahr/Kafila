package cluster

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// fakeStage stands in for a shard server. It records what it was asked to do so
// the façade's fan-out behaviour can be asserted without a running model.
type fakeStage struct {
	name string

	devices  []ml.DeviceID
	total    uint64
	vram     uint64
	ctxLen   int
	pid      int
	port     int
	exited   bool
	loadErr  error
	pingErr  error
	closeErr error

	loaded int
	pinged int
	closed int

	// emit is the stream Completion replays to its callback.
	emit []llm.CompletionResponse
	// delay is inserted before each emitted response, so trace spans have
	// something measurable in them.
	delay time.Duration
}

func (f *fakeStage) ModelPath() string { return "/models/" + f.name }

func (f *fakeStage) Load(context.Context, ml.SystemInfo, []ml.DeviceInfo, bool) ([]ml.DeviceID, error) {
	f.loaded++
	if f.loadErr != nil {
		return nil, f.loadErr
	}
	return f.devices, nil
}

func (f *fakeStage) Ping(context.Context) error {
	f.pinged++
	return f.pingErr
}

func (f *fakeStage) WaitUntilRunning(context.Context) error { return nil }

func (f *fakeStage) Completion(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	for _, r := range f.emit {
		if f.delay > 0 {
			time.Sleep(f.delay)
		}
		fn(r)
	}
	return nil
}

func (f *fakeStage) Chat(context.Context, llm.ChatRequest, func(llm.ChatResponse)) error { return nil }

func (f *fakeStage) ApplyChatTemplate(context.Context, llm.ChatRequest) (string, error) {
	return "templated:" + f.name, nil
}

func (f *fakeStage) Embedding(context.Context, string) ([]float32, int, error) {
	return []float32{1, 2, 3}, 3, nil
}

func (f *fakeStage) Tokenize(context.Context, string) ([]int, error) { return []int{1, 2}, nil }
func (f *fakeStage) Detokenize(context.Context, []int) (string, error) {
	return "detok:" + f.name, nil
}

func (f *fakeStage) Close() error {
	f.closed++
	return f.closeErr
}

func (f *fakeStage) MemorySize() (uint64, uint64) { return f.total, f.vram }

func (f *fakeStage) VRAMByGPU(id ml.DeviceID) uint64 {
	for _, d := range f.devices {
		if d == id {
			return f.vram
		}
	}
	return 0
}

func (f *fakeStage) Pid() int     { return f.pid }
func (f *fakeStage) GetPort() int { return f.port }

func (f *fakeStage) GetDeviceInfos(context.Context) []ml.DeviceInfo {
	var out []ml.DeviceInfo
	for _, d := range f.devices {
		out = append(out, ml.DeviceInfo{DeviceID: d, Name: f.name})
	}
	return out
}

func (f *fakeStage) HasExited() bool { return f.exited }
func (f *fakeStage) ContextLength() int {
	return f.ctxLen
}

var _ llm.LlamaServer = (*fakeStage)(nil)

func twoStagePipeline(t *testing.T, a, b *fakeStage) *Pipeline {
	t.Helper()

	r := NewRegistry()
	if err := r.Add(Node{Name: a.name}); err != nil {
		t.Fatal(err)
	}
	if err := r.Add(Node{Name: b.name}); err != nil {
		t.Fatal(err)
	}

	plan, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}

	p, err := NewPipeline(plan, []llm.LlamaServer{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func singleStagePipeline(t *testing.T, s *fakeStage, traceOut *bytes.Buffer) *Pipeline {
	t.Helper()

	r := NewRegistry()
	if err := r.Add(Node{Name: s.name}); err != nil {
		t.Fatal(err)
	}
	plan, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}

	// A typed nil *bytes.Buffer would be a non-nil io.Writer, so the nil case
	// has to stay untyped or tracing would never actually be off.
	var out io.Writer
	if traceOut != nil {
		out = traceOut
	}

	p, err := NewPipeline(plan, []llm.LlamaServer{s}, out)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestNewPipelineRejectsMismatchedStages(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(Node{Name: "a"})
	_ = r.Add(Node{Name: "b"})

	plan, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := NewPipeline(plan, []llm.LlamaServer{&fakeStage{name: "a"}}, nil); err == nil {
		t.Error("accepted one server for a two-stage plan")
	}
	if _, err := NewPipeline(plan, []llm.LlamaServer{&fakeStage{name: "a"}, nil}, nil); err == nil {
		t.Error("accepted a nil server")
	}
}

// Loading half a pipeline leaves memory held that the scheduler believes is
// free, so the first failure must stop the rest.
func TestLoadStopsAtFirstFailure(t *testing.T) {
	a := &fakeStage{name: "a", devices: []ml.DeviceID{{ID: "0", Library: "Metal"}}}
	b := &fakeStage{name: "b", loadErr: errors.New("out of memory")}
	p := twoStagePipeline(t, a, b)

	if _, err := p.Load(t.Context(), ml.SystemInfo{}, nil, false); err == nil {
		t.Fatal("Load reported success despite a failing stage")
	}
	if a.loaded != 1 || b.loaded != 1 {
		t.Errorf("loads: a=%d b=%d, want 1 each", a.loaded, b.loaded)
	}
}

func TestLoadUnionsDevices(t *testing.T) {
	shared := ml.DeviceID{ID: "0", Library: "Metal"}
	a := &fakeStage{name: "a", devices: []ml.DeviceID{shared}}
	b := &fakeStage{name: "b", devices: []ml.DeviceID{shared, {ID: "1", Library: "Metal"}}}
	p := twoStagePipeline(t, a, b)

	ids, err := p.Load(t.Context(), ml.SystemInfo{}, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Errorf("got %d device ids, want 2 deduplicated: %v", len(ids), ids)
	}
}

// Memory must sum: the scheduler decides what else fits from this number, so
// undercounting risks an out-of-memory load elsewhere.
func TestMemorySizeSums(t *testing.T) {
	a := &fakeStage{name: "a", total: 700, vram: 600}
	b := &fakeStage{name: "b", total: 300, vram: 250}
	p := twoStagePipeline(t, a, b)

	total, vram := p.MemorySize()
	if total != 1000 || vram != 850 {
		t.Errorf("MemorySize = %d/%d, want 1000/850", total, vram)
	}
}

func TestVRAMByGPUSumsSharedDevice(t *testing.T) {
	dev := ml.DeviceID{ID: "0", Library: "Metal"}
	a := &fakeStage{name: "a", devices: []ml.DeviceID{dev}, vram: 600}
	b := &fakeStage{name: "b", devices: []ml.DeviceID{dev}, vram: 250}
	p := twoStagePipeline(t, a, b)

	if got := p.VRAMByGPU(dev); got != 850 {
		t.Errorf("VRAMByGPU = %d, want 850", got)
	}
	if got := p.VRAMByGPU(ml.DeviceID{ID: "9", Library: "CUDA"}); got != 0 {
		t.Errorf("VRAMByGPU for an unknown device = %d, want 0", got)
	}
}

// A pipeline cannot serve a context longer than its most constrained stage.
func TestContextLengthIsTheMinimum(t *testing.T) {
	a := &fakeStage{name: "a", ctxLen: 8192}
	b := &fakeStage{name: "b", ctxLen: 4096}
	p := twoStagePipeline(t, a, b)

	if got := p.ContextLength(); got != 4096 {
		t.Errorf("ContextLength = %d, want 4096", got)
	}
}

// Any stage dying makes the pipeline unusable, so any exit is the pipeline's.
func TestHasExitedIfAnyStageDied(t *testing.T) {
	a := &fakeStage{name: "a"}
	b := &fakeStage{name: "b"}
	p := twoStagePipeline(t, a, b)

	if p.HasExited() {
		t.Error("reported exited with both stages alive")
	}
	b.exited = true
	if !p.HasExited() {
		t.Error("did not report exited with one stage dead")
	}
}

// A stage left running holds device memory the scheduler has written off, so
// Close must attempt every stage even after one fails.
func TestCloseAttemptsEveryStage(t *testing.T) {
	a := &fakeStage{name: "a", closeErr: errors.New("boom")}
	b := &fakeStage{name: "b"}
	p := twoStagePipeline(t, a, b)

	if err := p.Close(); err == nil {
		t.Error("Close hid a stage failure")
	}
	if a.closed != 1 || b.closed != 1 {
		t.Errorf("closes: a=%d b=%d, want 1 each", a.closed, b.closed)
	}
}

func TestPingChecksEveryStage(t *testing.T) {
	a := &fakeStage{name: "a"}
	b := &fakeStage{name: "b", pingErr: errors.New("unreachable")}
	p := twoStagePipeline(t, a, b)

	if err := p.Ping(t.Context()); err == nil {
		t.Error("Ping succeeded with an unreachable stage")
	} else if !strings.Contains(err.Error(), "b") {
		t.Errorf("error does not name the failing stage: %v", err)
	}
}

// Head-owned operations must not fan out; the head holds the tokenizer.
func TestHeadOwnedOperationsUseTheHead(t *testing.T) {
	a := &fakeStage{name: "head"}
	b := &fakeStage{name: "tail"}
	p := twoStagePipeline(t, a, b)

	if got := p.ModelPath(); got != "/models/head" {
		t.Errorf("ModelPath = %q, want the head's", got)
	}
	if got, _ := p.Detokenize(t.Context(), nil); got != "detok:head" {
		t.Errorf("Detokenize = %q, want the head's", got)
	}
	if got, _ := p.ApplyChatTemplate(t.Context(), llm.ChatRequest{}); got != "templated:head" {
		t.Errorf("ApplyChatTemplate = %q, want the head's", got)
	}
}

// Embeddings need the whole model. Returning a partial result silently would be
// worse than refusing.
func TestEmbeddingRefusedOnSplitPlan(t *testing.T) {
	p := twoStagePipeline(t, &fakeStage{name: "a"}, &fakeStage{name: "b"})
	if _, _, err := p.Embedding(t.Context(), "hello"); err == nil {
		t.Error("returned an embedding from a split plan")
	}

	solo := singleStagePipeline(t, &fakeStage{name: "solo"}, nil)
	if _, _, err := solo.Embedding(t.Context(), "hello"); err != nil {
		t.Errorf("single-node embedding failed: %v", err)
	}
}

func TestCompletionForwardsEveryResponse(t *testing.T) {
	s := &fakeStage{
		name: "solo",
		emit: []llm.CompletionResponse{
			{Content: "he"},
			{Content: "llo"},
			{Done: true, EvalCount: 2},
		},
	}
	p := singleStagePipeline(t, s, nil)

	var got []llm.CompletionResponse
	if err := p.Completion(t.Context(), llm.CompletionRequest{}, func(r llm.CompletionResponse) {
		got = append(got, r)
	}); err != nil {
		t.Fatal(err)
	}

	if len(got) != 3 {
		t.Fatalf("forwarded %d responses, want 3", len(got))
	}
	if got[0].Content != "he" || got[1].Content != "llo" || !got[2].Done {
		t.Errorf("responses altered in transit: %+v", got)
	}
}

// The trace must separate time-to-first-token from per-token decode, since the
// whole design predicts those behave differently under a split.
func TestCompletionTracesPrefillAndDecodeSeparately(t *testing.T) {
	var buf bytes.Buffer

	s := &fakeStage{
		name:  "solo",
		delay: 2 * time.Millisecond,
		emit: []llm.CompletionResponse{
			{Content: "a"},
			{Content: "b"},
			{Content: "c"},
			{Done: true},
		},
	}
	p := singleStagePipeline(t, s, &buf)

	if err := p.Completion(t.Context(), llm.CompletionRequest{}, func(llm.CompletionResponse) {}); err != nil {
		t.Fatal(err)
	}

	sums, err := trace.Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("trace is not parseable: %v", err)
	}
	if len(sums) != 1 {
		t.Fatalf("got %d requests in the trace, want 1", len(sums))
	}

	var sum *trace.Summary
	for _, v := range sums {
		sum = v
	}

	n := sum.Node("solo")
	if n == nil {
		t.Fatal("no spans recorded for the head node")
	}

	if n.ByPhase[trace.PhasePrefill] <= 0 {
		t.Error("no prefill time recorded; TTFT is not separated")
	}
	if n.ByPhase[trace.PhaseDecode] <= 0 {
		t.Error("no decode time recorded")
	}

	// One TTFT span, two inter-token spans, one final span.
	if n.Spans != 4 {
		t.Errorf("recorded %d spans, want 4", n.Spans)
	}
	if n.Overlapped {
		t.Error("wrapper spans overlapped; the residual is meaningless")
	}
}

// Tracing is optional and must not be load-bearing.
func TestCompletionWorksWithoutTracing(t *testing.T) {
	s := &fakeStage{name: "solo", emit: []llm.CompletionResponse{{Content: "x"}, {Done: true}}}
	p := singleStagePipeline(t, s, nil)

	calls := 0
	if err := p.Completion(t.Context(), llm.CompletionRequest{}, func(llm.CompletionResponse) {
		calls++
	}); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Errorf("forwarded %d responses with tracing off, want 2", calls)
	}
}

// Each request must land under its own id, or the dashboard cannot separate
// concurrent work.
func TestRequestIDsAreDistinct(t *testing.T) {
	var buf bytes.Buffer
	s := &fakeStage{name: "solo", emit: []llm.CompletionResponse{{Content: "x"}, {Done: true}}}
	p := singleStagePipeline(t, s, &buf)

	for range 3 {
		if err := p.Completion(t.Context(), llm.CompletionRequest{}, func(llm.CompletionResponse) {}); err != nil {
			t.Fatal(err)
		}
	}

	sums, err := trace.Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 3 {
		t.Errorf("got %d distinct requests, want 3", len(sums))
	}
}

func TestGetDeviceInfosDeduplicates(t *testing.T) {
	dev := ml.DeviceID{ID: "0", Library: "Metal"}
	a := &fakeStage{name: "a", devices: []ml.DeviceID{dev}}
	b := &fakeStage{name: "b", devices: []ml.DeviceID{dev}}
	p := twoStagePipeline(t, a, b)

	if got := p.GetDeviceInfos(t.Context()); len(got) != 1 {
		t.Errorf("got %d devices, want 1 after deduplication", len(got))
	}
}
