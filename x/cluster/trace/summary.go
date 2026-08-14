package trace

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"time"
)

// NodeSummary is what one node's clock can say about one request, and nothing
// more.
//
// Elapsed and Accounted both come from that node's monotonic clock, so their
// difference is exact. That difference is the point: it is time the node spent
// on the request that no span claimed, and publishing it keeps instrumentation
// gaps visible instead of letting them be silently absorbed into whichever
// measured bucket happens to be nearby.
type NodeSummary struct {
	Node string

	// Elapsed is the furthest offset any span reached — how long the request
	// was live on this node.
	Elapsed time.Duration

	// Accounted is the sum of span durations.
	Accounted time.Duration

	// Residual is Elapsed minus Accounted: measured-but-unattributed time.
	Residual time.Duration

	// Overlapped is true when Accounted exceeds Elapsed, which means spans on
	// this node overlapped in time. That is not an error, but it makes
	// Residual meaningless, so it is flagged rather than hidden.
	Overlapped bool

	ByKind  map[Kind]time.Duration
	ByPhase map[Phase]time.Duration

	Spans int
}

// Summary aggregates a request's spans per node. There is deliberately no
// cluster-wide "total time" derived from spans: nodes run concurrently during
// prefill, so summing across them would overcount, and comparing their clocks
// would be meaningless. Wall-clock total belongs to whoever served the request
// and is reported separately.
type Summary struct {
	Request string
	Nodes   []*NodeSummary
	Hops    []Hop

	// InconsistentHops counts hops whose sender and receiver measurements
	// disagree — the receiver claimed more time than the sender saw end to
	// end. Non-zero means the instrumentation is wrong, not the network.
	InconsistentHops int
}

// TotalHopBytes is the payload moved between nodes for this request.
func (s *Summary) TotalHopBytes() int64 {
	var n int64
	for _, h := range s.Hops {
		n += h.Bytes
	}
	return n
}

// InFlight is the total time genuinely in transit across all hops. Exact: it
// subtracts each receiver's self-reported work from the sender's round trip and
// makes no symmetry assumption.
func (s *Summary) InFlight() time.Duration {
	var d time.Duration
	for _, h := range s.Hops {
		d += h.InFlight()
	}
	return d
}

// Node returns a node's summary, or nil.
func (s *Summary) Node(name string) *NodeSummary {
	for _, n := range s.Nodes {
		if n.Node == name {
			return n
		}
	}
	return nil
}

// Parse reads an NDJSON span stream and aggregates it.
//
// Lines carrying a "hop" type are collected as hops; everything else is read as
// a span. Unparseable lines are an error rather than a silent skip: a trace with
// holes in it produces confidently wrong summaries.
func Parse(r io.Reader) (map[string]*Summary, error) {
	out := map[string]*Summary{}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), 8<<20)

	for line := 1; sc.Scan(); line++ {
		raw := sc.Bytes()
		if len(raw) == 0 {
			continue
		}

		var probe struct {
			Type    string `json:"type"`
			Request string `json:"request"`
		}
		if err := json.Unmarshal(raw, &probe); err != nil {
			return nil, fmt.Errorf("trace line %d: %w", line, err)
		}

		sum := out[probe.Request]
		if sum == nil {
			sum = &Summary{Request: probe.Request}
			out[probe.Request] = sum
		}

		if probe.Type == "hop" {
			var h Hop
			if err := json.Unmarshal(raw, &h); err != nil {
				return nil, fmt.Errorf("trace line %d: %w", line, err)
			}
			sum.Hops = append(sum.Hops, h)
			if !h.Consistent() {
				sum.InconsistentHops++
			}
			continue
		}

		var s Span
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, fmt.Errorf("trace line %d: %w", line, err)
		}
		sum.add(s)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	for _, sum := range out {
		sum.finish()
	}
	return out, nil
}

func (s *Summary) add(sp Span) {
	n := s.Node(sp.Node)
	if n == nil {
		n = &NodeSummary{
			Node:    sp.Node,
			ByKind:  map[Kind]time.Duration{},
			ByPhase: map[Phase]time.Duration{},
		}
		s.Nodes = append(s.Nodes, n)
	}

	n.Spans++
	n.Accounted += sp.Duration
	n.ByKind[sp.Kind] += sp.Duration
	n.ByPhase[sp.Phase] += sp.Duration
	if sp.Offset > n.Elapsed {
		n.Elapsed = sp.Offset
	}
}

func (s *Summary) finish() {
	for _, n := range s.Nodes {
		if n.Accounted > n.Elapsed {
			n.Overlapped = true
			n.Residual = 0
			continue
		}
		n.Residual = n.Elapsed - n.Accounted
	}
	sort.Slice(s.Nodes, func(i, j int) bool { return s.Nodes[i].Node < s.Nodes[j].Node })
}
