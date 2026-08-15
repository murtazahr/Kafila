// Package wire carries activations between pipeline stages.
//
// The payload is raw tensor bytes in the model's native dtype, with a small
// JSON header describing them. JSON alone is not an option: a prefill chunk is
// tokens × hidden × dtype bytes, which is 4 MiB for a 2048-token chunk on a
// 1024-wide bf16 model and four times that on a 4096-wide one. Base64 inside
// JSON would add a third to every hop and force the whole payload through an
// encoder on the critical path.
//
// Nothing here carries a wall-clock timestamp. Stages do not share a clock, and
// a receiver cannot say anything true about when a sender sent something. What
// it can say is how long it spent, so responses report a duration and the
// sender derives transit from its own round trip. See x/cluster/trace.
package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"time"
)

// MaxHeaderSize bounds the JSON header. Headers are a few hundred bytes; this
// is loose enough never to bind and tight enough that a corrupt length prefix
// cannot make the reader allocate wildly.
const MaxHeaderSize = 1 << 20

// MaxPayloadSize bounds the tensor payload at 256 MiB. A 2048-token prefill
// chunk is 4 MiB on the reference model and 16 MiB on a 4096-wide one, so this
// leaves two orders of magnitude of headroom while still refusing a garbage
// length rather than trying to allocate it.
const MaxPayloadSize = 256 << 20

// Phase mirrors trace.Phase without importing it, so the wire format does not
// depend on the shape of the telemetry package.
type Phase string

const (
	PhasePrefill Phase = "prefill"
	PhaseDecode  Phase = "decode"
	PhaseAlign   Phase = "align"
)

// StageReport is what one node contributes as a frame passes through it.
//
// The frame accumulates these on its way round the ring, so by the time it
// reaches the head it carries the whole circuit's account of itself. That is
// the only way to attribute time without a shared clock: each node measures
// itself on its own monotonic clock, and the head — which times the round trip
// on its own — subtracts the sum to learn what was in flight.
type StageReport struct {
	Node string `json:"node"`

	// Compute is how long this node spent in its blocks.
	Compute time.Duration `json:"compute_ns"`

	// CacheOffset is where this node's caches rest afterwards. Every node must
	// agree, since the offset drives RoPE and mask construction.
	CacheOffset int `json:"cache_offset"`

	// Simulated is delay this node was told to inject rather than time it
	// really spent. It is reported separately so a measurement is never
	// confused with a demonstration.
	Simulated time.Duration `json:"simulated_ns,omitempty"`
}

// Request is the header accompanying an activation travelling round the ring.
type Request struct {
	// Request identifies the inference request, so a trace can be reassembled
	// across stages by causality rather than by comparing clocks.
	Request string `json:"request"`

	Phase Phase `json:"phase"`

	// Token is the decode token index, or -1 outside decode. Chunk is the
	// prefill chunk index, or -1 outside prefill.
	Token int `json:"token"`
	Chunk int `json:"chunk"`

	// DType and Shape describe the payload. The dtype is carried explicitly
	// because it is the model's compute dtype, not something the receiver
	// should infer — getting it wrong reinterprets the bytes silently.
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`

	// SeqOffsets is each row's absolute position in its sequence. This is the
	// value that drives RoPE and mask construction, so every stage must use
	// the same one; disagreement produces plausible-looking wrong output
	// rather than an error.
	SeqOffsets []int32 `json:"seq_offsets"`

	// SeqQueryLens is each row's real query length. Values below the padded
	// length mark a tail that must be masked out.
	SeqQueryLens []int32 `json:"seq_query_lens"`

	// Seq identifies the circuit this frame belongs to. Requests are processed
	// one at a time, so this is a check rather than a router: a frame arriving
	// with the wrong sequence means the ring has lost its place, which would
	// otherwise show up as quietly wrong output.
	Seq uint64 `json:"seq"`

	// Reports accumulate as the frame travels. Each node appends its own
	// before forwarding.
	Reports []StageReport `json:"reports,omitempty"`

	// Reset asks every node to drop its cache state as the frame passes. Used
	// by alignment, which travels the same path as everything else so it
	// cannot take a different route and disagree.
	Reset bool `json:"reset,omitempty"`
}

// TotalCompute sums what every node reported spending in its blocks.
func (r Request) TotalCompute() time.Duration {
	var d time.Duration
	for _, s := range r.Reports {
		d += s.Compute
	}
	return d
}

// TotalSimulated sums delay that was injected rather than spent.
func (r Request) TotalSimulated() time.Duration {
	var d time.Duration
	for _, s := range r.Reports {
		d += s.Simulated
	}
	return d
}

// OffsetsAgree reports whether every node's caches rest at the same token, and
// returns the first disagreement it finds.
func (r Request) OffsetsAgree() (int, bool) {
	if len(r.Reports) == 0 {
		return 0, true
	}
	want := r.Reports[0].CacheOffset
	for _, s := range r.Reports[1:] {
		if s.CacheOffset != want {
			return s.CacheOffset, false
		}
	}
	return want, true
}

// Response is the header returned by a stage that has run its blocks.
type Response struct {
	Request string `json:"request"`

	// DType and Shape describe the returned activation.
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`

	// Duration is how long the responding stage spent, measured on its own
	// monotonic clock. The sender subtracts it from its round trip to get the
	// time genuinely in flight, which needs no shared clock and no assumption
	// about link symmetry.
	Duration time.Duration `json:"duration_ns"`

	// CacheOffset is the token offset this stage's caches now rest at. The
	// coordinator compares it across stages to detect the divergence that
	// would otherwise corrupt output silently.
	CacheOffset int `json:"cache_offset"`

	// Error carries a failure that happened after the response began, when
	// an HTTP status can no longer be set.
	Error string `json:"error,omitempty"`
}

// WriteFrame writes a header and payload as one framed message:
//
//	uint32  header length, big endian
//	uint32  payload length, big endian
//	bytes   JSON header
//	bytes   payload
//
// Both lengths precede both bodies so a reader can allocate exactly once and
// reject an implausible frame before reading any of it.
func WriteFrame(w io.Writer, header any, payload []byte) error {
	h, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("wire: marshal header: %w", err)
	}
	if len(h) > MaxHeaderSize {
		return fmt.Errorf("wire: header is %d bytes, limit is %d", len(h), MaxHeaderSize)
	}
	if len(payload) > MaxPayloadSize {
		return fmt.Errorf("wire: payload is %d bytes, limit is %d", len(payload), MaxPayloadSize)
	}

	var prefix [8]byte
	binary.BigEndian.PutUint32(prefix[0:4], uint32(len(h)))
	binary.BigEndian.PutUint32(prefix[4:8], uint32(len(payload)))

	if _, err := w.Write(prefix[:]); err != nil {
		return fmt.Errorf("wire: write prefix: %w", err)
	}
	if _, err := w.Write(h); err != nil {
		return fmt.Errorf("wire: write header: %w", err)
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return fmt.Errorf("wire: write payload: %w", err)
		}
	}
	return nil
}

// ReadFrame reads one framed message, decoding the header into out.
//
// A truncated frame is an error rather than a short read: a partial activation
// would be interpreted as a valid tensor of the wrong length, and the failure
// would surface as degraded output rather than as a fault.
func ReadFrame(r io.Reader, out any) ([]byte, error) {
	var prefix [8]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		return nil, fmt.Errorf("wire: read prefix: %w", err)
	}

	headerLen := binary.BigEndian.Uint32(prefix[0:4])
	payloadLen := binary.BigEndian.Uint32(prefix[4:8])

	if headerLen > MaxHeaderSize {
		return nil, fmt.Errorf("wire: header claims %d bytes, limit is %d", headerLen, MaxHeaderSize)
	}
	if payloadLen > MaxPayloadSize {
		return nil, fmt.Errorf("wire: payload claims %d bytes, limit is %d", payloadLen, MaxPayloadSize)
	}
	if headerLen == 0 {
		return nil, fmt.Errorf("wire: frame has no header")
	}

	h := make([]byte, headerLen)
	if _, err := io.ReadFull(r, h); err != nil {
		return nil, fmt.Errorf("wire: read header: %w", err)
	}
	if err := json.Unmarshal(h, out); err != nil {
		return nil, fmt.Errorf("wire: decode header: %w", err)
	}

	if payloadLen == 0 {
		return nil, nil
	}

	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("wire: read payload: %w", err)
	}
	return payload, nil
}

// PayloadSize returns the bytes an activation of this shape and dtype occupies,
// and reports false for a dtype whose width is unknown.
//
// Callers use it to check a header against its payload before trusting either.
// A shape that disagrees with the byte count means the two were built from
// different tensors, and reinterpreting the bytes anyway would produce a
// well-formed array of wrong values.
func PayloadSize(shape []int, itemSize int) (int, bool) {
	if itemSize <= 0 || len(shape) == 0 {
		return 0, false
	}
	n := itemSize
	for _, d := range shape {
		if d < 0 {
			return 0, false
		}
		n *= d
	}
	return n, true
}
