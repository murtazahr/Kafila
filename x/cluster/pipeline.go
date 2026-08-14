package cluster

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/cluster/trace"
)

// Pipeline fronts a plan's shards behind the interface Ollama's scheduler
// already speaks, so the split is invisible above it.
//
// Every inference request in the server funnels through Server.scheduleRunner,
// which returns an llm.LlamaServer and never sees anything more specific.
// Satisfying that interface is therefore the whole of what is required for the
// GUI, the CLI, the OpenAI-compatible routes and the registry to keep working
// unchanged — no compatibility shims anywhere above this line.
//
// The methods are written for an arbitrary number of stages even though Stage 1
// only ever builds one. A single-stage pipeline is the degenerate case, not a
// special case: it exercises planning, selection, tracing and this façade while
// inference itself stays byte-identical to the unsharded runner, so anything
// that breaks implicates the new machinery rather than the split.
type Pipeline struct {
	plan   *Plan
	stages []llm.LlamaServer

	// traceOut receives the NDJSON span stream. Nil disables recording; call
	// sites stay unconditional because a nil Recorder is inert.
	traceOut io.Writer

	requests atomic.Uint64
}

// Satisfying this interface is the entire compatibility requirement: the GUI,
// the CLI, the OpenAI and Anthropic routes and the registry all reach inference
// through it and nothing narrower. If this assertion holds, they keep working.
var _ llm.LlamaServer = (*Pipeline)(nil)

// NewPipeline binds a plan to the shard servers implementing its stages. The
// servers must be in pipeline order, one per stage.
func NewPipeline(plan *Plan, stages []llm.LlamaServer, traceOut io.Writer) (*Pipeline, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if len(stages) != plan.Len() {
		return nil, fmt.Errorf("cluster: %d shard servers for a %d-stage plan", len(stages), plan.Len())
	}
	for i, s := range stages {
		if s == nil {
			return nil, fmt.Errorf("cluster: stage %d (%s) has no server", i, plan.Stages[i].Node.Name)
		}
	}

	return &Pipeline{plan: plan, stages: stages, traceOut: traceOut}, nil
}

// Plan returns the plan this pipeline executes.
func (p *Pipeline) Plan() *Plan { return p.plan }

// head is the stage that owns the embedding, the output projection and the
// sampler, and therefore drives a request.
func (p *Pipeline) head() llm.LlamaServer { return p.stages[0] }

func (p *Pipeline) nextRequestID() string {
	return "req-" + strconv.FormatUint(p.requests.Add(1), 10)
}

func (p *Pipeline) recorder(request string) *trace.Recorder {
	if p.traceOut == nil {
		return nil
	}
	head := p.plan.Head()
	return trace.NewRecorder(p.traceOut, request, head.Node.Name, head.Blocks())
}

// --- request path -----------------------------------------------------------

// Completion runs a generation through the pipeline, recording where its time
// went.
//
// Spans are measured by this wrapper rather than taken from the engine's own
// reported durations. That is deliberate: wrapper spans partition the request's
// timeline exactly, with no overlap and no gaps that are really double counts,
// which is what makes the residual in a trace summary meaningful. The engine's
// figures are still forwarded to the caller untouched.
func (p *Pipeline) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	rec := p.recorder(p.nextRequestID())

	var (
		tokens int
		mark   = time.Now()
	)

	err := p.head().Completion(ctx, req, func(r llm.CompletionResponse) {
		now := time.Now()

		switch {
		case r.Content != "" && tokens == 0:
			// Time to first token: everything up to here is prompt work.
			rec.Record(trace.PhasePrefill, trace.KindCompute, now.Sub(mark),
				trace.Chunk(0), trace.Note("ttft"))
			mark, tokens = now, 1

		case r.Content != "":
			rec.Record(trace.PhaseDecode, trace.KindCompute, now.Sub(mark),
				trace.Token(tokens))
			mark, tokens = now, tokens+1
		}

		if r.Done {
			rec.Record(trace.PhaseDecode, trace.KindCompute, now.Sub(mark),
				trace.Token(tokens), trace.Note("done:"+r.DoneReason.String()))
			mark = now
		}

		fn(r)
	})

	if rec != nil {
		if rerr := rec.Err(); rerr != nil {
			// Losing a trace must never fail a request; surface it and move on.
			err = errors.Join(err, fmt.Errorf("cluster: trace write failed: %w", rerr))
		}
	}
	return err
}

// Chat delegates to the head, which owns the tokenizer and the template.
func (p *Pipeline) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return p.head().Chat(ctx, req, fn)
}

// ApplyChatTemplate delegates to the head.
func (p *Pipeline) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return p.head().ApplyChatTemplate(ctx, req)
}

// Embedding delegates to the head. Embeddings need the full model, so on a
// split plan this is the tail's output routed back — not yet implemented, and
// rejected rather than silently returning a partial result.
func (p *Pipeline) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	if !p.plan.IsSingleNode() {
		return nil, 0, errors.New("cluster: embeddings are not supported on a split plan yet")
	}
	return p.head().Embedding(ctx, input)
}

// Tokenize delegates to the head, which holds the tokenizer.
func (p *Pipeline) Tokenize(ctx context.Context, content string) ([]int, error) {
	return p.head().Tokenize(ctx, content)
}

// Detokenize delegates to the head.
func (p *Pipeline) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return p.head().Detokenize(ctx, tokens)
}

// --- lifecycle --------------------------------------------------------------

// Load brings every stage up and returns the union of the devices they claimed.
//
// A partial load is not a usable pipeline, so the first failure stops the rest:
// leaving half the stages resident would hold memory the scheduler believes is
// free.
func (p *Pipeline) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	rec := p.recorder(p.nextRequestID())

	var ids []ml.DeviceID
	seen := map[ml.DeviceID]bool{}

	for i, s := range p.stages {
		start := time.Now()
		got, err := s.Load(ctx, systemInfo, gpus, requireFull)
		rec.Record(trace.PhaseLoad, trace.KindCompute, time.Since(start),
			trace.Note(p.plan.Stages[i].Node.Name))
		if err != nil {
			return nil, fmt.Errorf("cluster: stage %d (%s): %w", i, p.plan.Stages[i].Node.Name, err)
		}
		for _, id := range got {
			if !seen[id] {
				seen[id] = true
				ids = append(ids, id)
			}
		}
	}

	return ids, nil
}

// WaitUntilRunning waits for every stage. A pipeline is ready only when all of
// it is.
func (p *Pipeline) WaitUntilRunning(ctx context.Context) error {
	for i, s := range p.stages {
		if err := s.WaitUntilRunning(ctx); err != nil {
			return fmt.Errorf("cluster: stage %d (%s): %w", i, p.plan.Stages[i].Node.Name, err)
		}
	}
	return nil
}

// Ping checks every stage. One unreachable node makes the pipeline unusable,
// so a partial answer would be misleading.
func (p *Pipeline) Ping(ctx context.Context) error {
	for i, s := range p.stages {
		if err := s.Ping(ctx); err != nil {
			return fmt.Errorf("cluster: stage %d (%s): %w", i, p.plan.Stages[i].Node.Name, err)
		}
	}
	return nil
}

// Close shuts every stage down, attempting all of them before reporting. A
// stage left running would hold device memory the scheduler has written off.
func (p *Pipeline) Close() error {
	var errs []error
	for i, s := range p.stages {
		if err := s.Close(); err != nil {
			errs = append(errs, fmt.Errorf("stage %d (%s): %w", i, p.plan.Stages[i].Node.Name, err))
		}
	}
	return errors.Join(errs...)
}

// HasExited reports whether any stage has died. The pipeline cannot serve a
// request without all of them, so any exit is the pipeline's exit.
func (p *Pipeline) HasExited() bool {
	for _, s := range p.stages {
		if s.HasExited() {
			return true
		}
	}
	return false
}

// --- accounting -------------------------------------------------------------

// MemorySize sums what the stages hold. The scheduler uses this to decide what
// else fits, so undercounting risks an out-of-memory load elsewhere.
func (p *Pipeline) MemorySize() (total, vram uint64) {
	for _, s := range p.stages {
		t, v := s.MemorySize()
		total += t
		vram += v
	}
	return total, vram
}

// VRAMByGPU sums a device's usage across stages. Stages on different machines
// report devices the local scheduler will never ask about, so those contribute
// nothing; stages sharing a device correctly add up.
func (p *Pipeline) VRAMByGPU(id ml.DeviceID) uint64 {
	var n uint64
	for _, s := range p.stages {
		n += s.VRAMByGPU(id)
	}
	return n
}

// GetDeviceInfos returns the devices across all stages, deduplicated by ID.
func (p *Pipeline) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	var out []ml.DeviceInfo
	seen := map[ml.DeviceID]bool{}
	for _, s := range p.stages {
		for _, d := range s.GetDeviceInfos(ctx) {
			if seen[d.DeviceID] {
				continue
			}
			seen[d.DeviceID] = true
			out = append(out, d)
		}
	}
	return out
}

// ContextLength is the smallest any stage can serve: a pipeline cannot process
// a context longer than its most constrained member.
func (p *Pipeline) ContextLength() int {
	n := 0
	for _, s := range p.stages {
		if c := s.ContextLength(); n == 0 || (c > 0 && c < n) {
			n = c
		}
	}
	return n
}

// ModelPath returns the head's, which is the model every stage sharded.
func (p *Pipeline) ModelPath() string { return p.head().ModelPath() }

// Pid returns the head's process id. The scheduler uses it to detect an
// orphaned runner, and the head is the process it talks to.
func (p *Pipeline) Pid() int { return p.head().Pid() }

// GetPort returns the head's port, which is where device discovery polls.
func (p *Pipeline) GetPort() int { return p.head().GetPort() }
