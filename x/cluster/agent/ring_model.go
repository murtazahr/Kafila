package agent

import (
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/shard"
	"github.com/ollama/ollama/x/tokenizer"
)

// RingModel presents a ring of shards as though the model were whole.
//
// It runs the head's blocks, dispatches the hidden state into the ring, and
// blocks until the frame comes back round. Because it satisfies base.Model, the
// runner's prefill and decode loops, its sampler and its detokenizer all work
// unmodified — they call Forward and Unembed and never learn that the blocks in
// between ran on other machines.
type RingModel struct {
	local  *Shard
	next   *Link
	ret    *RingReturn
	stages []StageInfo

	seq atomic.Uint64
	rec *trace.Recorder

	// timeout bounds a circuit. A ring has no reply path, so a node that dies
	// mid-frame would otherwise hang the request forever.
	timeout time.Duration
}

// StageInfo describes a node downstream, for reporting rather than for running.
type StageInfo struct {
	Name      string
	Blocks    string
	Address   string
	Role      string
	Simulated time.Duration
}

var (
	_ base.Model = (*RingModel)(nil)

	// A ring holds cache state on every node, so the runner must be told to
	// stop reusing prefixes it can only rewind on one of them.
	_ base.Distributed = (*RingModel)(nil)
)

// NewRingModel composes the head with the ring it dispatches into.
func NewRingModel(local *Shard, next *Link, ret *RingReturn, stages []StageInfo) (*RingModel, error) {
	if local == nil {
		return nil, fmt.Errorf("agent: ring needs a local head shard")
	}
	if !local.Assignment.Role.IsHead() {
		return nil, fmt.Errorf("agent: local shard %s is not the head", local.Blocks())
	}
	if next == nil || ret == nil {
		return nil, fmt.Errorf("agent: ring needs both a next link and a return listener")
	}
	return &RingModel{local: local, next: next, ret: ret, stages: stages, timeout: 5 * time.Minute}, nil
}

// Trace attaches a recorder.
func (r *RingModel) Trace(rec *trace.Recorder) { r.rec = rec }

// Stages describes the nodes downstream.
func (r *RingModel) Stages() []StageInfo { return r.stages }

// Blocks is the head's own range.
func (r *RingModel) Blocks() shard.Range { return r.local.Blocks() }

// Forward runs the head's blocks and sends the result round the ring.
//
// The head times the whole circuit on its own clock and every node reports its
// own compute on its own. Subtracting one from the other gives the time
// genuinely in flight, without any comparison between clocks.
func (r *RingModel) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	phase, token, chunk := classify(b)

	started := time.Now()
	local, err := r.local.Forward(b)
	if err != nil {
		slog.Error("ring: head forward failed", "error", err)
		return nil, nil
	}
	headCompute := time.Since(started)
	r.rec.Record(trace.Phase(phase), trace.KindCompute, headCompute,
		trace.Token(token), trace.Chunk(chunk))

	payload, err := local.Bytes()
	if err != nil {
		slog.Error("ring: could not serialize the hidden state", "error", err)
		return nil, nil
	}

	req := wire.Request{
		Phase: phase, Token: token, Chunk: chunk,
		DType: local.DType().String(), Shape: local.Dims(),
		SeqOffsets: b.SeqOffsets, SeqQueryLens: b.SeqQueryLens,
		Seq: r.seq.Add(1),
	}

	out, err := r.circuit(req, payload, headCompute)
	if err != nil {
		slog.Error("ring: circuit failed", "error", err)
		return nil, nil
	}

	h, err := rebuild(out.req, out.payload)
	if err != nil {
		slog.Error("ring: could not rebuild the returned hidden state", "error", err)
		return nil, nil
	}
	return h, h
}

// circuit dispatches a frame and waits for it to come back round, recording
// what the trip cost.
func (r *RingModel) circuit(req wire.Request, payload []byte, headCompute time.Duration) (returned, error) {
	// Register before dispatching: a fast ring can deliver before a later
	// registration would have been made.
	wait := r.ret.Expect(req.Seq)

	started := time.Now()
	if err := r.next.Send(req, payload); err != nil {
		r.ret.Forget(req.Seq)
		return returned{}, err
	}

	select {
	case out := <-wait:
		elapsed := time.Since(started)
		r.report(req, out.req, elapsed, headCompute, len(payload))
		return out, out.err

	case <-time.After(r.timeout):
		r.ret.Forget(req.Seq)
		return returned{}, fmt.Errorf("agent: circuit %d did not return within %s", req.Seq, r.timeout)
	}
}

// report records what the circuit cost, split between what nodes said they
// spent and what was left over.
func (r *RingModel) report(sent, back wire.Request, elapsed time.Duration, headCompute time.Duration, bytes int) {
	compute := back.TotalCompute() + headCompute

	// The head's own outgoing link is not in the frame's reports — nothing
	// appends on its behalf — so it is added here. Without it the circuit
	// would appear to have one fewer leg than it has.
	simulated := back.TotalSimulated() + r.next.Simulated()

	// Whatever the circuit took that no node claimed is time on the wire. It
	// is exact: one clock measured the whole trip, and every subtraction is
	// from a duration rather than a timestamp.
	inFlight := elapsed - compute
	if inFlight < 0 {
		inFlight = 0
	}

	r.rec.RecordHop(trace.Hop{
		To:    "ring",
		Phase: trace.Phase(sent.Phase),
		Token: sent.Token, Chunk: sent.Chunk,
		RoundTrip:      elapsed,
		RemoteDuration: compute,
		Simulated:      simulated,
		Symmetric:      true,
		Bytes:          int64(bytes),
		Stages:         r.stageDurations(back, headCompute),
	})
}

// stageDurations renders the circuit node by node, head first, so a reader can
// follow the frame the way it actually travelled.
func (r *RingModel) stageDurations(req wire.Request, headCompute time.Duration) []trace.StageTime {
	out := make([]trace.StageTime, 0, len(req.Reports)+1)
	out = append(out, trace.StageTime{
		Node:        r.local.Blocks().String(),
		Compute:     headCompute,
		Simulated:   r.next.Simulated(),
		CacheOffset: r.local.Offset(),
	})
	for _, s := range req.Reports {
		out = append(out, trace.StageTime{
			Node: s.Node, Compute: s.Compute, Simulated: s.Simulated, CacheOffset: s.CacheOffset,
		})
	}
	return out
}

// Unembed projects to logits on the head, which owns the output projection.
func (r *RingModel) Unembed(x *mlx.Array) *mlx.Array {
	out, err := r.local.Unembed(x)
	if err != nil {
		slog.Error("ring: unembed failed", "error", err)
		return nil
	}
	return out
}

// LoadWeights has no meaning for a composed model; see PipelineModel.
func (r *RingModel) LoadWeights(map[string]*mlx.Array) error {
	return fmt.Errorf("agent: a ring is composed from loaded shards, not loaded from tensors")
}

// NewCaches returns the head's. Every other node keeps its own.
func (r *RingModel) NewCaches() []cache.Cache { return r.local.Model.NewCaches() }

// Tokenizer returns the head's.
func (r *RingModel) Tokenizer() *tokenizer.Tokenizer { return r.local.Model.Tokenizer() }

// MaxContextLength is the model's, identical on every node.
func (r *RingModel) MaxContextLength() int { return r.local.Model.MaxContextLength() }

// Align sends an alignment frame round the ring and checks every node agrees
// on where its caches rest.
//
// Alignment travels the same path as everything else rather than over a
// side-channel, so it cannot take a different route and reach a different
// conclusion. If any node disagrees, a second frame goes round telling all of
// them to reset: not every cache kind can rewind, so reprocessing is the
// fallback that always works.
func (r *RingModel) Align() error {
	out, err := r.circuit(wire.Request{
		Phase: wire.PhaseAlign, Token: -1, Chunk: -1, Seq: r.seq.Add(1),
	}, nil, 0)
	if err != nil {
		return err
	}

	offset, agreed := out.req.OffsetsAgree()
	if agreed && offset == r.local.Offset() {
		return nil
	}

	slog.Warn("ring: cache offsets disagree; flushing every node",
		"head", r.local.Offset(), "ring", offset)
	r.rec.Record(trace.PhaseAlign, trace.KindCompute, 0,
		trace.Note("offsets diverged; flushing every node"))

	return r.Reset()
}

// Reset clears cache state everywhere.
func (r *RingModel) Reset() error {
	if err := r.local.Reset(); err != nil {
		return err
	}
	_, err := r.circuit(wire.Request{
		Phase: wire.PhaseAlign, Token: -1, Chunk: -1, Reset: true, Seq: r.seq.Add(1),
	}, nil, 0)
	return err
}
