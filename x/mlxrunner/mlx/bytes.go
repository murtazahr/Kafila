package mlx

/*
#include "generated.h"

// cgo has no representation for the 2-byte float types these two accessors
// return, so the cast to an untyped pointer is done in C. Everything else has
// a Go-representable element type and is called directly.
static const void *ollamaArrayDataFloat16(const mlx_array arr) {
	return (const void *)mlx_array_data_float16(arr);
}

static const void *ollamaArrayDataBFloat16(const mlx_array arr) {
	return (const void *)mlx_array_data_bfloat16(arr);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// This file adds dtype-preserving access to an array's raw element storage.
//
// Ints and Floats already read element data, but each panics unless the array
// is exactly I32 or F32. Hidden states carry the model's compute dtype, which
// for these models is BF16, so moving one through those accessors means an
// AsType(DTypeFloat32) conversion that doubles its size. For activations
// crossing a network that doubling is paid on every hop, so the transport path
// needs a route that keeps the original dtype.

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

// dataPointer returns a pointer to the array's contiguous element storage.
//
// The pointer is taken through the accessor matching the array's own dtype
// rather than a single type-punned call, because the C API validates the dtype
// on each accessor.
func (t *Array) dataPointer() (unsafe.Pointer, error) {
	switch dt := t.DType(); dt {
	case DTypeBool:
		return unsafe.Pointer(C.mlx_array_data_bool(t.ctx)), nil
	case DTypeUint8:
		return unsafe.Pointer(C.mlx_array_data_uint8(t.ctx)), nil
	case DTypeUint16:
		return unsafe.Pointer(C.mlx_array_data_uint16(t.ctx)), nil
	case DTypeUint32:
		return unsafe.Pointer(C.mlx_array_data_uint32(t.ctx)), nil
	case DTypeUint64:
		return unsafe.Pointer(C.mlx_array_data_uint64(t.ctx)), nil
	case DTypeInt8:
		return unsafe.Pointer(C.mlx_array_data_int8(t.ctx)), nil
	case DTypeInt16:
		return unsafe.Pointer(C.mlx_array_data_int16(t.ctx)), nil
	case DTypeInt32:
		return unsafe.Pointer(C.mlx_array_data_int32(t.ctx)), nil
	case DTypeInt64:
		return unsafe.Pointer(C.mlx_array_data_int64(t.ctx)), nil
	case DTypeFloat16:
		return unsafe.Pointer(C.ollamaArrayDataFloat16(t.ctx)), nil
	case DTypeFloat32:
		return unsafe.Pointer(C.mlx_array_data_float32(t.ctx)), nil
	case DTypeFloat64:
		return unsafe.Pointer(C.mlx_array_data_float64(t.ctx)), nil
	case DTypeBFloat16:
		return unsafe.Pointer(C.ollamaArrayDataBFloat16(t.ctx)), nil
	case DTypeComplex64:
		return unsafe.Pointer(C.mlx_array_data_complex64(t.ctx)), nil
	default:
		return nil, fmt.Errorf("mlx: no data accessor for dtype %v", dt)
	}
}

// RawBytes returns the array's element storage as a byte slice that aliases
// MLX's own memory, in the array's native dtype and without copying.
//
// The array is evaluated first: MLX arrays are lazy, and reading the data
// pointer of an unevaluated array races its evaluation and yields garbage.
//
// The returned slice stays valid only while MLX retains the underlying buffer,
// which Sweep is free to release. Callers must either Pin the array for the
// lifetime of the slice or copy out of it before the next Sweep. Use Bytes
// unless the copy is worth avoiding.
func (t *Array) RawBytes() ([]byte, error) {
	Eval(t)

	n := t.NumBytes()
	if n == 0 {
		return nil, nil
	}

	p, err := t.dataPointer()
	if err != nil {
		return nil, err
	}
	if p == nil {
		return nil, fmt.Errorf("mlx: array %q has no backing storage", t.name)
	}

	return unsafe.Slice((*byte)(p), n), nil
}

// Bytes returns a copy of the array's element storage in its native dtype.
//
// Unlike RawBytes the result has no lifetime tie to MLX, so it survives Sweep
// and is safe to hand to a writer that outlives the calling frame.
func (t *Array) Bytes() ([]byte, error) {
	raw, err := t.RawBytes()
	if err != nil {
		return nil, err
	}
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
