package wire

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

func TestFrameRoundTrip(t *testing.T) {
	want := Request{
		Request:      "req-7",
		Phase:        PhasePrefill,
		Token:        -1,
		Chunk:        3,
		DType:        "BF16",
		Shape:        []int{1, 2048, 1024},
		SeqOffsets:   []int32{6144},
		SeqQueryLens: []int32{2048},
	}
	payload := bytes.Repeat([]byte{0xAB, 0xCD}, 1024)

	var buf bytes.Buffer
	if err := WriteFrame(&buf, want, payload); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}

	var got Request
	gotPayload, err := ReadFrame(&buf, &got)
	if err != nil {
		t.Fatalf("ReadFrame: %v", err)
	}

	if got.Request != want.Request || got.Phase != want.Phase || got.Chunk != want.Chunk {
		t.Errorf("header identity = %+v, want %+v", got, want)
	}
	if got.DType != want.DType {
		t.Errorf("dtype = %q, want %q", got.DType, want.DType)
	}
	if len(got.Shape) != 3 || got.Shape[1] != 2048 || got.Shape[2] != 1024 {
		t.Errorf("shape = %v, want %v", got.Shape, want.Shape)
	}
	if len(got.SeqOffsets) != 1 || got.SeqOffsets[0] != 6144 {
		t.Errorf("seq offsets = %v, want %v", got.SeqOffsets, want.SeqOffsets)
	}
	if !bytes.Equal(gotPayload, payload) {
		t.Errorf("payload changed in transit: %d bytes vs %d", len(gotPayload), len(payload))
	}
	if buf.Len() != 0 {
		t.Errorf("%d bytes left unread after the frame", buf.Len())
	}
}

// Activations are bf16. A frame must move those bytes untouched — any widening
// or reinterpretation en route defeats the point of the byte path.
func TestFramePreservesArbitraryBytes(t *testing.T) {
	payload := make([]byte, 512)
	for i := range payload {
		payload[i] = byte(i * 7)
	}

	var buf bytes.Buffer
	if err := WriteFrame(&buf, Request{Request: "r", DType: "BF16", Shape: []int{256}}, payload); err != nil {
		t.Fatal(err)
	}

	var got Request
	back, err := ReadFrame(&buf, &got)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(back, payload) {
		t.Error("payload bytes were altered")
	}
}

// Several frames share one stream, so each must consume exactly its own bytes.
func TestFramesAreSequential(t *testing.T) {
	var buf bytes.Buffer

	for i := range 3 {
		h := Request{Request: "r", Token: i, DType: "BF16", Shape: []int{4}}
		if err := WriteFrame(&buf, h, []byte{byte(i), byte(i), byte(i), byte(i), byte(i), byte(i), byte(i), byte(i)}); err != nil {
			t.Fatal(err)
		}
	}

	for i := range 3 {
		var got Request
		payload, err := ReadFrame(&buf, &got)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if got.Token != i {
			t.Errorf("frame %d has token %d", i, got.Token)
		}
		if len(payload) != 8 || payload[0] != byte(i) {
			t.Errorf("frame %d payload = %v", i, payload)
		}
	}

	if buf.Len() != 0 {
		t.Errorf("%d bytes left over", buf.Len())
	}
}

func TestFrameWithoutPayload(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteFrame(&buf, Response{Request: "r", Duration: time.Millisecond}, nil); err != nil {
		t.Fatal(err)
	}

	var got Response
	payload, err := ReadFrame(&buf, &got)
	if err != nil {
		t.Fatal(err)
	}
	if payload != nil {
		t.Errorf("expected no payload, got %d bytes", len(payload))
	}
	if got.Duration != time.Millisecond {
		t.Errorf("duration = %s, want 1ms", got.Duration)
	}
}

// A truncated frame must fail. A partial activation would otherwise be read as
// a valid tensor of the wrong length, surfacing as degraded output rather than
// as a fault.
func TestTruncatedFrameFails(t *testing.T) {
	var full bytes.Buffer
	if err := WriteFrame(&full, Request{Request: "r", DType: "BF16", Shape: []int{128}}, make([]byte, 256)); err != nil {
		t.Fatal(err)
	}

	for name, n := range map[string]int{
		"prefix only":     4,
		"header cut":      12,
		"payload missing": full.Len() - 100,
		"one byte short":  full.Len() - 1,
	} {
		t.Run(name, func(t *testing.T) {
			var got Request
			if _, err := ReadFrame(bytes.NewReader(full.Bytes()[:n]), &got); err == nil {
				t.Error("ReadFrame accepted a truncated frame")
			}
		})
	}
}

// A corrupt length prefix must be refused rather than allocated.
func TestImplausibleLengthsRejected(t *testing.T) {
	for name, prefix := range map[string][]byte{
		"huge header":  {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0},
		"huge payload": {0, 0, 0, 2, 0xFF, 0xFF, 0xFF, 0xFF},
		"zero header":  {0, 0, 0, 0, 0, 0, 0, 4},
	} {
		t.Run(name, func(t *testing.T) {
			var got Request
			if _, err := ReadFrame(bytes.NewReader(prefix), &got); err == nil {
				t.Error("ReadFrame accepted an implausible length")
			}
		})
	}
}

func TestOversizeWritesRejected(t *testing.T) {
	var buf bytes.Buffer

	big := Request{Request: strings.Repeat("x", MaxHeaderSize+1)}
	if err := WriteFrame(&buf, big, nil); err == nil {
		t.Error("WriteFrame accepted an oversize header")
	}

	if err := WriteFrame(io.Discard, Request{Request: "r"}, make([]byte, MaxPayloadSize+1)); err == nil {
		t.Error("WriteFrame accepted an oversize payload")
	}
}

func TestUnmarshalableHeaderRejected(t *testing.T) {
	if err := WriteFrame(io.Discard, make(chan int), nil); err == nil {
		t.Error("WriteFrame accepted a header that cannot be marshalled")
	}
}

func TestPayloadSize(t *testing.T) {
	for name, tc := range map[string]struct {
		shape    []int
		itemSize int
		want     int
		ok       bool
	}{
		"decode token bf16":  {[]int{1, 1, 1024}, 2, 2048, true},
		"prefill chunk bf16": {[]int{1, 2048, 1024}, 2, 4194304, true},
		"f32":                {[]int{1, 1, 1024}, 4, 4096, true},
		"scalar-ish":         {[]int{1}, 2, 2, true},
		"zero dim":           {[]int{1, 0, 1024}, 2, 0, true},
		"no shape":           {nil, 2, 0, false},
		"unknown dtype":      {[]int{1, 1024}, 0, 0, false},
		"negative dim":       {[]int{1, -1}, 2, 0, false},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := PayloadSize(tc.shape, tc.itemSize)
			if ok != tc.ok {
				t.Fatalf("ok = %v, want %v", ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("PayloadSize = %d, want %d", got, tc.want)
			}
		})
	}
}

// A header whose shape disagrees with the payload it arrived with means the two
// came from different tensors. Reinterpreting the bytes anyway would yield a
// well-formed array of wrong values, so callers are expected to check.
func TestShapeAndPayloadCanBeCrossChecked(t *testing.T) {
	h := Request{DType: "BF16", Shape: []int{1, 4, 1024}}

	want, ok := PayloadSize(h.Shape, 2)
	if !ok {
		t.Fatal("PayloadSize failed on a valid shape")
	}
	if want != 8192 {
		t.Fatalf("expected 8192 bytes, got %d", want)
	}

	var buf bytes.Buffer
	if err := WriteFrame(&buf, h, make([]byte, want/2)); err != nil {
		t.Fatal(err)
	}

	var got Request
	payload, err := ReadFrame(&buf, &got)
	if err != nil {
		t.Fatal(err)
	}

	size, _ := PayloadSize(got.Shape, 2)
	if len(payload) == size {
		t.Error("a mismatched payload went undetected by the size check")
	}
}

// The wire format must carry no absolute timestamp: stages do not share a
// clock, and a field a reader could subtract across machines invites exactly
// the skew error the trace package exists to prevent.
func TestResponseCarriesDurationNotTimestamp(t *testing.T) {
	raw, err := json.Marshal(Response{Request: "r", Duration: 5 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}

	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}

	for _, banned := range []string{"time", "timestamp", "ts", "sent_at", "received_at", "started", "wall"} {
		if _, ok := fields[banned]; ok {
			t.Errorf("response carries %q; stages must not compare clocks", banned)
		}
	}
	if _, ok := fields["duration_ns"]; !ok {
		t.Error("response is missing duration_ns")
	}
}
