package trace

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

func TestRecorderWritesNDJSON(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{Start: 0, End: 14})

	r.Record(PhasePrefill, KindCompute, 5*time.Millisecond, Chunk(0), Bytes(4<<20))
	r.Record(PhaseDecode, KindCompute, 900*time.Microsecond, Token(0))

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2:\n%s", len(lines), buf.String())
	}

	var first Span
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("line 1 is not valid JSON: %v", err)
	}

	if first.Request != "req-1" || first.Node != "node-a" {
		t.Errorf("identity = %s/%s, want req-1/node-a", first.Request, first.Node)
	}
	if first.Blocks != "[0,14)" {
		t.Errorf("blocks = %q, want [0,14)", first.Blocks)
	}
	if first.Chunk != 0 || first.Token != NoIndex {
		t.Errorf("indices = chunk %d token %d, want chunk 0 token %d", first.Chunk, first.Token, NoIndex)
	}
	if first.Duration != 5*time.Millisecond {
		t.Errorf("duration = %s, want 5ms", first.Duration)
	}
	if first.Bytes != 4<<20 {
		t.Errorf("bytes = %d, want %d", first.Bytes, 4<<20)
	}
}

// A span must carry no absolute timestamp. Any field a reader could subtract
// across nodes invites exactly the clock-skew error this package exists to
// prevent, so the wire format is asserted directly.
func TestSpanCarriesNoAbsoluteTimestamp(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})
	r.Record(PhaseDecode, KindCompute, time.Millisecond, Token(3))

	var raw map[string]any
	if err := json.Unmarshal(buf.Bytes(), &raw); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}

	for _, banned := range []string{"time", "timestamp", "ts", "started", "started_at", "wall", "clock"} {
		if _, ok := raw[banned]; ok {
			t.Errorf("span carries %q; absolute timestamps must not cross nodes", banned)
		}
	}
	for _, want := range []string{"offset_ns", "duration_ns"} {
		if _, ok := raw[want]; !ok {
			t.Errorf("span is missing %q", want)
		}
	}
}

// Seq must be dense and monotonic per node so a reader can order spans without
// consulting a clock.
func TestRecorderSeqIsMonotonic(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})
	for range 5 {
		r.Record(PhaseDecode, KindCompute, time.Microsecond)
	}

	for i, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var s Span
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			t.Fatalf("line %d: %v", i, err)
		}
		if s.Seq != i {
			t.Errorf("line %d has seq %d", i, s.Seq)
		}
	}
}

func TestRecorderIsConcurrencySafe(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})

	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.Record(PhaseDecode, KindCompute, time.Duration(i)*time.Microsecond, Token(i))
		}()
	}
	wg.Wait()

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 50 {
		t.Fatalf("got %d lines, want 50", len(lines))
	}

	seen := map[int]bool{}
	for _, line := range lines {
		var s Span
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			t.Fatalf("interleaved write corrupted a line: %v", err)
		}
		if seen[s.Seq] {
			t.Errorf("duplicate seq %d", s.Seq)
		}
		seen[s.Seq] = true
	}
}

// A nil recorder must stay usable so instrumentation at call sites can be
// unconditional.
func TestNilRecorderIsInert(t *testing.T) {
	var r *Recorder
	r.Record(PhaseDecode, KindCompute, time.Second)
	r.RecordHop(Hop{})

	ran := false
	if d := r.Time(PhaseDecode, KindCompute, func() { ran = true }); d != 0 {
		t.Errorf("nil recorder returned duration %s", d)
	}
	if !ran {
		t.Error("nil recorder skipped the function it was asked to time")
	}
	if err := r.Err(); err != nil {
		t.Errorf("nil recorder reported %v", err)
	}
}

func TestTimeMeasuresTheFunction(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})

	d := r.Time(PhaseDecode, KindCompute, func() { time.Sleep(5 * time.Millisecond) })
	if d < 4*time.Millisecond {
		t.Errorf("measured %s for a 5ms sleep", d)
	}

	var s Span
	if err := json.Unmarshal(buf.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	if s.Duration != d {
		t.Errorf("recorded %s but returned %s", s.Duration, d)
	}
}

func TestHopInFlight(t *testing.T) {
	for name, tc := range map[string]struct {
		rtt, remote time.Duration
		want        time.Duration
		consistent  bool
	}{
		"normal":            {10 * time.Millisecond, 6 * time.Millisecond, 4 * time.Millisecond, true},
		"no remote work":    {10 * time.Millisecond, 0, 10 * time.Millisecond, true},
		"remote equals rtt": {10 * time.Millisecond, 10 * time.Millisecond, 0, true},
		"remote exceeds":    {10 * time.Millisecond, 12 * time.Millisecond, 0, false},
	} {
		t.Run(name, func(t *testing.T) {
			h := Hop{RoundTrip: tc.rtt, RemoteDuration: tc.remote}
			if got := h.InFlight(); got != tc.want {
				t.Errorf("InFlight = %s, want %s", got, tc.want)
			}
			if got := h.Consistent(); got != tc.consistent {
				t.Errorf("Consistent = %v, want %v", got, tc.consistent)
			}
		})
	}
}

// Halving a round trip is only defensible on a symmetric link. Without that
// assertion the caller must get no number rather than a plausible wrong one.
func TestOneWayRefusesWithoutSymmetry(t *testing.T) {
	h := Hop{RoundTrip: 10 * time.Millisecond, RemoteDuration: 2 * time.Millisecond}

	if _, ok := h.OneWay(); ok {
		t.Error("OneWay produced an estimate for a link not asserted symmetric")
	}

	h.Symmetric = true
	got, ok := h.OneWay()
	if !ok {
		t.Fatal("OneWay refused on a symmetric link")
	}
	if want := 4 * time.Millisecond; got != want {
		t.Errorf("OneWay = %s, want %s", got, want)
	}
}

func TestParseAggregatesPerNode(t *testing.T) {
	var buf bytes.Buffer

	head := NewRecorder(&buf, "req-1", "head", shard.Range{Start: 0, End: 14})
	head.Record(PhasePrefill, KindCompute, 10*time.Millisecond, Chunk(0))
	head.Record(PhasePrefill, KindSerialize, 2*time.Millisecond, Chunk(0), Bytes(4<<20))

	tail := NewRecorder(&buf, "req-1", "tail", shard.Range{Start: 14, End: 28})
	tail.Record(PhasePrefill, KindDeserialize, time.Millisecond, Chunk(0))
	tail.Record(PhasePrefill, KindCompute, 9*time.Millisecond, Chunk(0))

	head.RecordHop(Hop{
		To: "tail", Phase: PhasePrefill, Chunk: 0,
		RoundTrip: 8 * time.Millisecond, RemoteDuration: 6 * time.Millisecond,
		Symmetric: true, Bytes: 4 << 20,
	})

	sums, err := Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	sum := sums["req-1"]
	if sum == nil {
		t.Fatal("no summary for req-1")
	}
	if len(sum.Nodes) != 2 {
		t.Fatalf("got %d nodes, want 2", len(sum.Nodes))
	}
	if sum.Nodes[0].Node != "head" {
		t.Errorf("nodes not sorted: first is %q", sum.Nodes[0].Node)
	}

	h := sum.Node("head")
	if h.Accounted != 12*time.Millisecond {
		t.Errorf("head accounted %s, want 12ms", h.Accounted)
	}
	if h.ByKind[KindCompute] != 10*time.Millisecond {
		t.Errorf("head compute %s, want 10ms", h.ByKind[KindCompute])
	}
	if h.Spans != 2 {
		t.Errorf("head spans %d, want 2", h.Spans)
	}

	if len(sum.Hops) != 1 {
		t.Fatalf("got %d hops, want 1", len(sum.Hops))
	}
	if got := sum.InFlight(); got != 2*time.Millisecond {
		t.Errorf("in-flight %s, want 2ms", got)
	}
	if got := sum.TotalHopBytes(); got != 4<<20 {
		t.Errorf("hop bytes %d, want %d", got, 4<<20)
	}
	if sum.InconsistentHops != 0 {
		t.Errorf("flagged %d inconsistent hops", sum.InconsistentHops)
	}
}

// Unattributed time must be published, not absorbed. This is the check that
// keeps the instrumentation honest as spans are added or missed.
func TestResidualIsReported(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})

	// Leave a real gap the spans do not describe.
	time.Sleep(20 * time.Millisecond)
	r.Record(PhaseDecode, KindCompute, time.Millisecond, Token(0))

	sums, err := Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	n := sums["req-1"].Node("node-a")
	if n.Overlapped {
		t.Fatal("single sequential span reported as overlapped")
	}
	if n.Residual < 15*time.Millisecond {
		t.Errorf("residual %s, expected to capture the ~19ms of unmeasured time", n.Residual)
	}
	if n.Residual+n.Accounted != n.Elapsed {
		t.Errorf("residual %s + accounted %s != elapsed %s", n.Residual, n.Accounted, n.Elapsed)
	}
}

// Overlapping spans make the residual meaningless. It must be flagged rather
// than reported as a negative or a zero that looks like full coverage.
func TestOverlapIsFlagged(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "node-a", shard.Range{})
	r.Record(PhaseDecode, KindCompute, time.Hour)

	sums, err := Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	n := sums["req-1"].Node("node-a")
	if !n.Overlapped {
		t.Error("an hour of spans inside a millisecond of elapsed time was not flagged")
	}
	if n.Residual != 0 {
		t.Errorf("residual %s reported despite overlap", n.Residual)
	}
}

func TestParseFlagsInconsistentHop(t *testing.T) {
	var buf bytes.Buffer
	r := NewRecorder(&buf, "req-1", "head", shard.Range{})
	r.RecordHop(Hop{
		To: "tail", RoundTrip: 5 * time.Millisecond, RemoteDuration: 9 * time.Millisecond,
	})

	sums, err := Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if sums["req-1"].InconsistentHops != 1 {
		t.Error("a receiver claiming more time than the sender observed was not flagged")
	}
}

// A malformed line must fail loudly: a trace with holes produces confidently
// wrong summaries.
func TestParseRejectsMalformedLines(t *testing.T) {
	in := strings.Join([]string{
		`{"request":"r","node":"a","duration_ns":1,"offset_ns":2}`,
		`{not json`,
	}, "\n")

	if _, err := Parse(strings.NewReader(in)); err == nil {
		t.Error("Parse accepted a malformed line")
	}
}

func TestParseSeparatesRequests(t *testing.T) {
	var buf bytes.Buffer
	NewRecorder(&buf, "req-1", "a", shard.Range{}).Record(PhaseDecode, KindCompute, time.Millisecond)
	NewRecorder(&buf, "req-2", "a", shard.Range{}).Record(PhaseDecode, KindCompute, 2*time.Millisecond)

	sums, err := Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(sums) != 2 {
		t.Fatalf("got %d requests, want 2", len(sums))
	}
	if got := sums["req-2"].Node("a").Accounted; got != 2*time.Millisecond {
		t.Errorf("req-2 accounted %s, want 2ms", got)
	}
}
