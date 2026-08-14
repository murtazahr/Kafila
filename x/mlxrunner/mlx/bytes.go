package mlx

// #include "generated.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// This file adds dtype-preserving access to an array's raw element storage.
//
// Ints and Floats already read element data, but each panics unless the array
// is exactly I32 or F32. Hidden states carry the model's compute dtype, which
// for these models is bf16, so moving one through those accessors means an
// AsType(DTypeFloat32) conversion that doubles its size. For activations
// crossing a network that doubling is paid on every hop, so the transport path
// needs a route that keeps the original dtype.
//
// The read goes through an unsigned integer view of the same width rather than
// the accessor matching the array's own dtype. That is not a stylistic choice:
// mlx-c guards its 16-bit float accessors behind HAS_FLOAT16 and HAS_BFLOAT16,
// which Apple clang defines and gcc does not, so libmlxc.so built for CUDA
// exports twelve data accessors where the Metal build exports fourteen —
// mlx_array_data_bfloat16 and mlx_array_data_float16 are simply absent. Because
// the symbols are resolved through dlopen into function pointers, calling one
// that was never defined jumps to address zero and takes the process with it.
//
// Reading every dtype through uint8/16/32/64, which are present everywhere,
// removes that whole class of failure and leaves one code path to test rather
// than one per platform.

// ItemSize returns the size in bytes of a single element of this dtype.
func (t DType) ItemSize() int {
	switch t {
	case DTypeBool, DTypeUint8, DTypeInt8:
		return 1
	case DTypeUint16, DTypeInt16, DTypeFloat16, DTypeBFloat16:
		return 2
	case DTypeUint32, DTypeInt32, DTypeFloat32:
		return 4
	case DTypeUint64, DTypeInt64, DTypeFloat64, DTypeComplex64:
		return 8
	default:
		return 0
	}
}

// byteView returns the unsigned integer dtype with the same width, which is the
// dtype an array is reinterpreted as in order to read its bytes.
func byteView(size int) (DType, bool) {
	switch size {
	case 1:
		return DTypeUint8, true
	case 2:
		return DTypeUint16, true
	case 4:
		return DTypeUint32, true
	case 8:
		return DTypeUint64, true
	default:
		return 0, false
	}
}

// rawPointer evaluates the array, reinterprets it as an unsigned integer of the
// same width, and returns a pointer to its storage.
//
// The returned array is the one that owns the pointer and must stay reachable
// for as long as the pointer is used.
func (t *Array) rawPointer() (*Array, unsafe.Pointer, error) {
	dt := t.DType()

	view, ok := byteView(dt.ItemSize())
	if !ok {
		return nil, nil, fmt.Errorf("mlx: cannot read bytes of dtype %v", dt)
	}

	v := t
	if dt != view {
		v = t.View(view)
	}

	// MLX is lazy; reading a data pointer before evaluation races the
	// evaluation and yields garbage.
	Eval(v)

	var p unsafe.Pointer
	switch view {
	case DTypeUint8:
		p = unsafe.Pointer(C.mlx_array_data_uint8(v.ctx))
	case DTypeUint16:
		p = unsafe.Pointer(C.mlx_array_data_uint16(v.ctx))
	case DTypeUint32:
		p = unsafe.Pointer(C.mlx_array_data_uint32(v.ctx))
	case DTypeUint64:
		p = unsafe.Pointer(C.mlx_array_data_uint64(v.ctx))
	}
	if p == nil {
		return nil, nil, fmt.Errorf("mlx: array %q has no backing storage", t.name)
	}

	return v, p, nil
}

// RawBytes returns the array's element storage as a byte slice aliasing MLX's
// own memory, in the array's native dtype and without copying.
//
// The slice is valid only until release is called, which the caller must do
// exactly once. Use Bytes unless the copy is genuinely worth avoiding; this
// exists for the transport path, where a prefill chunk can be megabytes.
func (t *Array) RawBytes() (data []byte, release func(), err error) {
	n := t.NumBytes()
	if n == 0 {
		return nil, func() {}, nil
	}

	v, p, err := t.rawPointer()
	if err != nil {
		return nil, nil, err
	}

	// Pin the owner so a Sweep between here and the caller's release cannot
	// free the memory the slice points into.
	Pin(v)
	return unsafe.Slice((*byte)(p), n), func() { Unpin(v) }, nil
}

// Bytes returns a copy of the array's element storage in its native dtype.
//
// Unlike RawBytes the result has no lifetime tie to MLX, so it survives Sweep
// and is safe to hand to a writer that outlives the calling frame.
func (t *Array) Bytes() ([]byte, error) {
	raw, release, err := t.RawBytes()
	if err != nil {
		return nil, err
	}
	defer release()

	if raw == nil {
		return nil, nil
	}

	out := make([]byte, len(raw))
	copy(out, raw)
	return out, nil
}

// FromBytes builds an array from raw element bytes in the given dtype and
// shape, the inverse of Bytes.
//
// This is the counterpart FromValues cannot provide: Go has no float16 or
// bfloat16 type, so those dtypes are unreachable through a typed slice and can
// only be reconstructed from their raw bytes.
//
// MLX copies the data, so the caller keeps ownership of the slice.
func FromBytes(data []byte, dtype DType, shape ...int) (*Array, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("mlx: FromBytes requires a shape")
	}

	itemSize := dtype.ItemSize()
	if itemSize == 0 {
		return nil, fmt.Errorf("mlx: FromBytes does not support dtype %v", dtype)
	}

	elements := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("mlx: FromBytes got negative dimension %d in shape %v", d, shape)
		}
		elements *= d
	}

	if want := elements * itemSize; len(data) != want {
		return nil, fmt.Errorf("mlx: FromBytes got %d bytes, shape %v of %v needs %d", len(data), shape, dtype, want)
	}

	if elements == 0 {
		return nil, fmt.Errorf("mlx: FromBytes got an empty shape %v", shape)
	}

	cShape := make([]C.int, len(shape))
	for i := range shape {
		cShape[i] = C.int(shape[i])
	}

	t := New("")
	t.ctx = C.mlx_array_new_data(
		unsafe.Pointer(&data[0]),
		unsafe.SliceData(cShape),
		C.int(len(cShape)),
		C.mlx_dtype(dtype),
	)
	return t, nil
}
