package mlx

import (
	"bytes"
	"testing"
)

// requireMLX skips a test when the MLX dynamic library is not loadable, so the
// package's pure-Go tests stay runnable on a machine without a built runtime.
func requireMLX(t *testing.T) {
	t.Helper()
	if err := CheckInit(); err != nil {
		t.Skipf("MLX runtime unavailable: %v", err)
	}
}

func TestDTypeItemSize(t *testing.T) {
	for name, tc := range map[string]struct {
		dtype DType
		want  int
	}{
		"bool":      {DTypeBool, 1},
		"uint8":     {DTypeUint8, 1},
		"int8":      {DTypeInt8, 1},
		"uint16":    {DTypeUint16, 2},
		"int16":     {DTypeInt16, 2},
		"float16":   {DTypeFloat16, 2},
		"bfloat16":  {DTypeBFloat16, 2},
		"uint32":    {DTypeUint32, 4},
		"int32":     {DTypeInt32, 4},
		"float32":   {DTypeFloat32, 4},
		"uint64":    {DTypeUint64, 8},
		"int64":     {DTypeInt64, 8},
		"float64":   {DTypeFloat64, 8},
		"complex64": {DTypeComplex64, 8},
	} {
		t.Run(name, func(t *testing.T) {
			if got := tc.dtype.ItemSize(); got != tc.want {
				t.Errorf("%v.ItemSize() = %d, want %d", tc.dtype, got, tc.want)
			}
		})
	}
}

func TestFromBytesValidation(t *testing.T) {
	for name, tc := range map[string]struct {
		data  []byte
		dtype DType
		shape []int
	}{
		"no shape":       {make([]byte, 16), DTypeFloat32, nil},
		"short data":     {make([]byte, 8), DTypeFloat32, []int{4}},
		"long data":      {make([]byte, 32), DTypeFloat32, []int{4}},
		"zero dimension": {nil, DTypeFloat32, []int{0, 4}},
		"negative dim":   {make([]byte, 16), DTypeFloat32, []int{-1, 4}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := FromBytes(tc.data, tc.dtype, tc.shape...); err == nil {
				t.Errorf("FromBytes accepted invalid input")
			}
		})
	}
}

func TestBytesRoundTripFloat32(t *testing.T) {
	requireMLX(t)

	want := []float32{1, -2.5, 3.25, 0, 1e-8, -1e8}
	src := FromValues(want, 2, 3)

	raw, err := src.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}
	if got, want := len(raw), len(want)*4; got != want {
		t.Fatalf("Bytes returned %d bytes, want %d", got, want)
	}

	dst, err := FromBytes(raw, DTypeFloat32, 2, 3)
	if err != nil {
		t.Fatalf("FromBytes failed: %v", err)
	}

	got := dst.Floats()
	if len(got) != len(want) {
		t.Fatalf("round trip produced %d elements, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("element %d = %v, want %v", i, got[i], want[i])
		}
	}

	if d := dst.Dims(); len(d) != 2 || d[0] != 2 || d[1] != 3 {
		t.Errorf("round trip shape = %v, want [2 3]", d)
	}
}

// The transport path exists to carry BF16 without converting to F32, so a
// BF16 array must survive a byte round trip bit-for-bit.
func TestBytesRoundTripBFloat16(t *testing.T) {
	requireMLX(t)

	// Values chosen to be exactly representable in bfloat16 so the comparison
	// after the round trip tests the transport rather than rounding.
	src := FromValues([]float32{1, -2, 0.5, 256, -0.125, 3}, 2, 3).AsType(DTypeBFloat16)
	Eval(src)

	if dt := src.DType(); dt != DTypeBFloat16 {
		t.Fatalf("source dtype = %v, want BF16", dt)
	}

	raw, err := src.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}
	if got, want := len(raw), 6*2; got != want {
		t.Fatalf("BF16 Bytes returned %d bytes, want %d", got, want)
	}

	dst, err := FromBytes(raw, DTypeBFloat16, 2, 3)
	if err != nil {
		t.Fatalf("FromBytes failed: %v", err)
	}
	if dt := dst.DType(); dt != DTypeBFloat16 {
		t.Fatalf("round trip dtype = %v, want BF16", dt)
	}

	again, err := dst.Bytes()
	if err != nil {
		t.Fatalf("Bytes on the round-tripped array failed: %v", err)
	}
	if !bytes.Equal(raw, again) {
		t.Errorf("BF16 round trip changed the bytes:\n got %x\nwant %x", again, raw)
	}

	// AsType is lazy, and Floats reads the data pointer without evaluating, so
	// both conversions must be materialized before they are read.
	wantF32, gotF32 := src.AsType(DTypeFloat32), dst.AsType(DTypeFloat32)
	Eval(wantF32, gotF32)

	wantVals, gotVals := wantF32.Floats(), gotF32.Floats()
	for i := range wantVals {
		if gotVals[i] != wantVals[i] {
			t.Errorf("element %d = %v, want %v", i, gotVals[i], wantVals[i])
		}
	}
}

// This is the measurement that justifies the whole file: a BF16 hidden state
// must cross the wire at half the size of the F32 conversion that Floats would
// have forced. Every activation hop pays this difference.
func TestBytesAvoidsFloat32Widening(t *testing.T) {
	requireMLX(t)

	const elements = 4096

	vals := make([]float32, elements)
	for i := range vals {
		vals[i] = float32(i%7) - 3
	}

	bf16 := FromValues(vals, 1, elements).AsType(DTypeBFloat16)
	Eval(bf16)

	native, err := bf16.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}

	widened := bf16.AsType(DTypeFloat32)
	Eval(widened)
	viaFloats := len(widened.Floats()) * 4

	if len(native) != elements*2 {
		t.Errorf("BF16 payload = %d bytes, want %d", len(native), elements*2)
	}
	if viaFloats != 2*len(native) {
		t.Errorf("F32 payload = %d bytes, expected twice the BF16 payload of %d", viaFloats, len(native))
	}

	t.Logf("hidden state of %d elements: %d bytes native BF16 vs %d bytes via F32", elements, len(native), viaFloats)
}

// RawBytes aliases MLX memory rather than copying, so it must report the same
// contents as Bytes while sharing storage with the array.
func TestRawBytesMatchesBytes(t *testing.T) {
	requireMLX(t)

	src := FromValues([]float32{1, 2, 3, 4}, 2, 2)

	raw, release, err := src.RawBytes()
	if err != nil {
		t.Fatalf("RawBytes failed: %v", err)
	}
	defer release()
	cp, err := src.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}

	if !bytes.Equal(raw, cp) {
		t.Errorf("RawBytes and Bytes disagree:\n raw %x\ncopy %x", raw, cp)
	}
	if len(raw) != src.NumBytes() {
		t.Errorf("RawBytes length %d != NumBytes %d", len(raw), src.NumBytes())
	}
}

// Bytes must copy: mutating the returned slice must not corrupt the array,
// since a caller holding the result across a Sweep would otherwise be writing
// into freed MLX storage.
func TestBytesIsACopy(t *testing.T) {
	requireMLX(t)

	src := FromValues([]float32{1, 2, 3, 4}, 2, 2)

	cp, err := src.Bytes()
	if err != nil {
		t.Fatalf("Bytes failed: %v", err)
	}
	for i := range cp {
		cp[i] = 0xFF
	}

	if got := src.Floats(); got[0] != 1 || got[3] != 4 {
		t.Errorf("mutating the Bytes result changed the array: %v", got)
	}
}
