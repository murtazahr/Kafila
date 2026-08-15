package agent

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/shard"
	"github.com/ollama/ollama/x/tokenizer"
)

// PipelineModel presents a split model as though it were whole.
//
// It runs the head's blocks locally, hands the hidden state to each downstream
// stage in turn, and returns what comes back. Because it satisfies base.Model,
// the runner's prefill and decode loops, its sampler, and its detokenizer all
// work unmodified — they call Forward and Unembed and never learn that the
// blocks in between ran on other machines.
//
// That is the whole trick, and it is why the split needed so little of the
// runner to change: the seam was already there.
type PipelineModel struct {
	local  *Shard
	stages []*Stage
	rec    *trace.Recorder
}

var _ base.Model = (*PipelineModel)(nil)

// NewPipelineModel composes the head shard with its downstream stages, in
// pipeline order.
func NewPipelineModel(local *Shard, stages []*Stage) (*PipelineModel, error) {
	if local == nil {
		return nil, fmt.Errorf("agent: pipeline needs a local head shard")
	}
	if !local.Assignment.Role.IsHead() {
		return nil, fmt.Errorf("agent: local shard %s is not the head", local.Blocks())
	}
	return &PipelineModel{local: local, stages: stages}, nil
}

// Trace attaches a recorder to the pipeline and to every stage.
func (p *PipelineModel) Trace(rec *trace.Recorder) {
	p.rec = rec
	for _, s := range p.stages {
		s.Trace(rec)
	}
}

// Stages returns the downstream stages, in order.
func (p *PipelineModel) Stages() []*Stage { return p.stages }

// Blocks returns the head's own block range.
func (p *PipelineModel) Blocks() shard.Range { return p.local.Blocks() }

// Forward runs the head's blocks, then each downstream stage.
//
// Phase and index come from the batch's shape rather than being threaded
// through: a query length above one is prefill, a single token is decode. That
// keeps the runner's loops untouched at the cost of an inference the trace
// makes visible anyway.
func (p *PipelineModel) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	phase, token, chunk := classify(b)

	local, err := p.local.Forward(b)
	if err != nil {
		slog.Error("pipeline: head forward failed", "error", err)
		return nil, nil
	}
	h := local

	for _, s := range p.stages {
		out, err := s.Forward(b, h, phase, token, chunk)
		if err != nil {
			// The runner's Forward signature has nowhere to put an error. A
			// nil hidden state stops the request in the caller, and the cause
			// is logged rather than swallowed.
			slog.Error("pipeline: stage forward failed", "stage", s.Name, "blocks", s.Blocks, "error", err)
			return nil, nil
		}
		h = out
	}

	return h, h
}

// Unembed projects to logits on the head, which owns the output projection.
func (p *PipelineModel) Unembed(x *mlx.Array) *mlx.Array {
	out, err := p.local.Unembed(x)
	if err != nil {
		slog.Error("pipeline: unembed failed", "error", err)
		return nil
	}
	return out
}

// LoadWeights is part of base.Model but has no meaning here: a pipeline is
// composed from shards that have already bound their own weights, so there is
// no set of tensors that belongs to it. Reaching this means something tried to
// load the pipeline as though it were an ordinary model, which would silently
// produce a model with no blocks.
func (p *PipelineModel) LoadWeights(map[string]*mlx.Array) error {
	return fmt.Errorf("agent: a pipeline is composed from loaded shards, not loaded from tensors")
}

// NewCaches returns the head's caches. Every other stage keeps its own, which
// is why they are not visible here: a cache belongs to the blocks that write
// it, and those blocks live elsewhere.
func (p *PipelineModel) NewCaches() []cache.Cache { return p.local.Model.NewCaches() }

// Tokenizer returns the head's, which is the one the runner uses.
func (p *PipelineModel) Tokenizer() *tokenizer.Tokenizer { return p.local.Model.Tokenizer() }

// MaxContextLength is the model's, identical on every stage.
func (p *PipelineModel) MaxContextLength() int { return p.local.Model.MaxContextLength() }

// Align makes every stage agree on where its caches rest.
//
// The invariant is that all stages sit at the same token offset, because the
// offset drives RoPE and mask construction and a disagreement produces
// plausible-looking wrong output rather than an error. The head owns the
// decision: it collects each stage's offset, and if any differs it resets all
// of them and starts over. Resetting rather than rewinding is deliberate — not
// every cache kind can rewind, so reprocessing is the fallback that always
// works.
func (p *PipelineModel) Align() error {
	want := p.local.Offset()

	agreed := true
	for _, s := range p.stages {
		got, err := s.Align(false)
		if err != nil {
			return err
		}
		if got != want {
			slog.Warn("pipeline: cache offsets disagree",
				"stage", s.Name, "stage_offset", got, "head_offset", want)
			agreed = false
		}
	}
	if agreed {
		return nil
	}

	p.rec.Record(trace.PhaseAlign, trace.KindCompute, 0, trace.Note("offsets diverged; flushing every stage"))

	if err := p.local.Reset(); err != nil {
		return err
	}
	for _, s := range p.stages {
		if _, err := s.Align(true); err != nil {
			return err
		}
	}
	return nil
}

// Reset clears cache state everywhere, so the next request starts clean.
func (p *PipelineModel) Reset() error {
	if err := p.local.Reset(); err != nil {
		return err
	}
	for _, s := range p.stages {
		if _, err := s.Align(true); err != nil {
			return err
		}
	}
	return nil
}

// classify infers the phase from the batch. A query length above one is a
// prefill chunk; a single token is decode.
func classify(b *batch.Batch) (wire.Phase, int, int) {
	n := 0
	if len(b.SeqQueryLens) > 0 {
		n = int(b.SeqQueryLens[0])
	}
	offset := 0
	if len(b.SeqOffsets) > 0 {
		offset = int(b.SeqOffsets[0])
	}

	if n > 1 {
		return wire.PhasePrefill, -1, offset
	}
	return wire.PhaseDecode, offset, -1
}
