package trace

import (
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// Recorder collects spans for one request on one node and writes them as
// NDJSON.
//
// One line per span, one file per run: the same stream feeds the dashboard and
// the benchmark, so there is no second format to drift out of step with what is
// actually measured.
//
// A Recorder is safe for concurrent use. Its clock is captured at construction
// and every Offset is relative to that, so a recorder is only meaningful for
// the request it was made for.
type Recorder struct {
	request string
	node    string
	blocks  string

	mu    sync.Mutex
	w     io.Writer
	enc   *json.Encoder
	start time.Time
	seq   int
	err   error
}

// NewRecorder starts recording a request on a node. A nil writer discards
// everything, so instrumentation can stay unconditional at call sites.
func NewRecorder(w io.Writer, request, node string, blocks shard.Range) *Recorder {
	r := &Recorder{
		request: request,
		node:    node,
		start:   time.Now(),
		w:       w,
	}
	if blocks.Len() > 0 {
		r.blocks = blocks.String()
	}
	if w != nil {
		r.enc = json.NewEncoder(w)
	}
	return r
}

// Err reports the first write error, if any. Recording failures never
// interrupt inference; they surface here instead.
func (r *Recorder) Err() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

// Elapsed is time since the recorder was created, on this node's clock.
func (r *Recorder) Elapsed() time.Duration {
	if r == nil {
		return 0
	}
	return time.Since(r.start)
}

// Record emits one span of the given duration.
func (r *Recorder) Record(phase Phase, kind Kind, d time.Duration, opts ...Option) {
	if r == nil {
		return
	}

	s := Span{
		Request:  r.request,
		Node:     r.node,
		Phase:    phase,
		Kind:     kind,
		Token:    NoIndex,
		Chunk:    NoIndex,
		Blocks:   r.blocks,
		Duration: d,
	}
	for _, o := range opts {
		o(&s)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	s.Seq = r.seq
	r.seq++
	// Offset marks where the span ended, which is what the recorder can know
	// without the caller telling it when the work began.
	s.Offset = time.Since(r.start)

	r.write(s)
}

// Time measures fn and records a span for it. This is the form call sites
// should prefer: it cannot report a duration that was not actually measured.
func (r *Recorder) Time(phase Phase, kind Kind, fn func(), opts ...Option) time.Duration {
	if r == nil {
		fn()
		return 0
	}
	start := time.Now()
	fn()
	d := time.Since(start)
	r.Record(phase, kind, d, opts...)
	return d
}

// RecordHop emits a hop. The caller is responsible for having measured
// RoundTrip on its own clock and for setting Symmetric honestly.
func (r *Recorder) RecordHop(h Hop) {
	if r == nil {
		return
	}
	h.Request = r.request
	if h.From == "" {
		h.From = r.node
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.write(struct {
		Type string `json:"type"`
		Hop
	}{Type: "hop", Hop: h})
}

// write assumes the lock is held.
func (r *Recorder) write(v any) {
	if r.enc == nil {
		return
	}
	if err := r.enc.Encode(v); err != nil && r.err == nil {
		r.err = err
	}
}

// Option adjusts a span before it is written.
type Option func(*Span)

// Token attaches a decode token index.
func Token(i int) Option { return func(s *Span) { s.Token = i } }

// Chunk attaches a prefill chunk index.
func Chunk(i int) Option { return func(s *Span) { s.Chunk = i } }

// Bytes attaches a payload size.
func Bytes(n int64) Option { return func(s *Span) { s.Bytes = n } }

// Note attaches a short human-readable detail.
func Note(text string) Option { return func(s *Span) { s.Note = text } }
