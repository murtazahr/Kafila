// Package trace records what a distributed request spent its time on, in a
// form that stays honest across machines that do not share a clock.
//
// The constraint that shapes everything here: NTP on commodity hardware leaves
// tens of milliseconds of skew, while a decode hop between two nodes is
// sub-millisecond. Subtracting one node's timestamp from another's therefore
// measures clock offset, not transit, and can produce negative transit times.
// Any figure derived that way is worthless, and worse, looks plausible.
//
// So a span carries no absolute timestamp at all. It carries a duration
// measured on one node's monotonic clock, and an offset from the start of the
// request on that same clock. Cross-node ordering comes from causality — token
// index, sequence number, and the hop edges between nodes — never from
// comparing clocks. The one genuinely cross-machine quantity, wire time, is
// measured as a sender-side round trip and halved under an assumption that is
// recorded as a field rather than left implicit.
package trace

import (
	"fmt"
	"time"
)

// Phase is the part of a request a span belongs to.
type Phase string

const (
	PhaseLoad    Phase = "load"    // model load and shard weight binding
	PhaseAlign   Phase = "align"   // the cache-offset alignment round trip
	PhasePrefill Phase = "prefill" // prompt processing, chunked
	PhaseDecode  Phase = "decode"  // token generation
)

// Kind is what a span was doing. Splitting these apart is the point: at
// sub-millisecond decode hops, serialization can exceed network time, and
// conflating them misattributes the cost of the whole design.
type Kind string

const (
	// KindCompute is time spent in the model's forward pass.
	KindCompute Kind = "compute"
	// KindSerialize is turning an activation into bytes.
	KindSerialize Kind = "serialize"
	// KindDeserialize is turning bytes back into an activation.
	KindDeserialize Kind = "deserialize"
	// KindWire is time on the network, derived from a sender-side round trip.
	KindWire Kind = "wire"
	// KindQueue is time a request waited before a node started it.
	KindQueue Kind = "queue"
	// KindWait is a node idle awaiting an upstream stage — the pipeline
	// bubble. Expected near-zero in steady-state prefill and near-total
	// during decode; measuring it is the direct check on that prediction.
	KindWait Kind = "wait"
)

// NoIndex marks a span that does not belong to a particular token or chunk.
const NoIndex = -1

// Span is one measured interval on one node.
//
// Duration and Offset are both from that node's monotonic clock and are exact.
// There is deliberately no absolute timestamp field: see the package comment.
type Span struct {
	Request string `json:"request"`
	Node    string `json:"node"`

	Phase Phase `json:"phase"`
	Kind  Kind  `json:"kind"`

	// Token is the index of the token being generated, or NoIndex outside
	// decode. Chunk is the prefill chunk index, or NoIndex outside prefill.
	Token int `json:"token"`
	Chunk int `json:"chunk"`

	// Seq orders spans within a single node and request. Cross-node ordering
	// must come from causality, not from comparing Seq across nodes.
	Seq int `json:"seq"`

	// Blocks is the transformer block range this node was responsible for,
	// rendered as a half-open interval. Empty for spans not tied to a shard.
	Blocks string `json:"blocks,omitempty"`

	// Offset is time since the request began on this node's clock. Safe to
	// compare only against other Offsets from the same node and request.
	Offset time.Duration `json:"offset_ns"`

	// Duration is the measured interval. Exact.
	Duration time.Duration `json:"duration_ns"`

	// Bytes is the payload size where the span moved data.
	Bytes int64 `json:"bytes,omitempty"`

	// Note carries a short human-readable detail, such as why a cache flush
	// happened. Not parsed.
	Note string `json:"note,omitempty"`
}

// Hop is one node-to-node transfer, measured entirely on the sender's clock.
//
// Deriving one-way time from a round trip is an assumption, not a measurement.
// Recording Symmetric as a field rather than dividing silently means a reader
// can see which figures depend on it, and an asymmetric link can be reported
// honestly instead of producing a confidently wrong number.
type Hop struct {
	Request string `json:"request"`
	From    string `json:"from"`
	To      string `json:"to"`

	Phase Phase `json:"phase"`
	Token int   `json:"token"`
	Chunk int   `json:"chunk"`

	// RoundTrip is send-to-acknowledgement, on the sender's monotonic clock.
	RoundTrip time.Duration `json:"round_trip_ns"`

	// RemoteDuration is what the receiver reported spending on its own clock,
	// as a duration rather than a timestamp. Subtracting it from RoundTrip
	// leaves the part that was genuinely in flight.
	RemoteDuration time.Duration `json:"remote_duration_ns"`

	// Symmetric records the assumption that the two directions cost the same.
	// When false, OneWay refuses to guess.
	Symmetric bool `json:"symmetric"`

	Bytes int64 `json:"bytes,omitempty"`
}

// InFlight is the round trip with the receiver's own work removed: the time
// genuinely spent in transit, both directions combined. Exact — it involves no
// cross-clock arithmetic and no symmetry assumption.
//
// Returns zero if the receiver reported spending longer than the whole round
// trip, which means the two measurements disagree and something is wrong with
// the instrumentation rather than with the network.
func (h Hop) InFlight() time.Duration {
	if h.RemoteDuration > h.RoundTrip {
		return 0
	}
	return h.RoundTrip - h.RemoteDuration
}

// OneWay is the estimated cost of a single direction, and reports false when
// that estimate is not defensible.
//
// It halves InFlight, which is only meaningful if the link is symmetric. A
// caller that has not asserted symmetry gets no number rather than a wrong one.
func (h Hop) OneWay() (time.Duration, bool) {
	if !h.Symmetric {
		return 0, false
	}
	return h.InFlight() / 2, true
}

// Consistent reports whether the two independently measured durations agree
// well enough to trust the hop. A receiver claiming more time than the sender
// observed end to end means the clocks or the instrumentation are wrong.
func (h Hop) Consistent() bool { return h.RemoteDuration <= h.RoundTrip }

func (h Hop) String() string {
	return fmt.Sprintf("%s→%s %s rtt=%s remote=%s inflight=%s",
		h.From, h.To, h.Phase, h.RoundTrip, h.RemoteDuration, h.InFlight())
}
