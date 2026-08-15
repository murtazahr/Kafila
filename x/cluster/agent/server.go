package agent

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"time"

	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Server exposes one shard over TCP.
//
// The protocol is deliberately small: a framed request carrying a hidden state
// and the positions it belongs at, a framed response carrying the hidden state
// this stage produced. Everything else a stage needs — which blocks it owns,
// which ends of the pipeline it holds — was decided at load and does not travel
// per request.
//
// Requests are served one at a time. A shard is a single model with a single
// set of caches, and the caches advance with every forward, so concurrent
// requests would interleave their positions and corrupt each other. The
// serialization is the same one the MLX runner already imposes, made explicit.
type Server struct {
	shard *Shard
	ln    net.Listener
}

// Listen binds a server for a shard. Address may be host:port or :port; use
// 127.0.0.1:0 to let the OS choose.
func Listen(s *Shard, address string) (*Server, error) {
	ln, err := net.Listen("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("agent: listen on %s: %w", address, err)
	}
	return &Server{shard: s, ln: ln}, nil
}

// Addr is the address the server accepted on, useful when the port was chosen
// by the OS.
func (s *Server) Addr() string { return s.ln.Addr().String() }

// Close stops accepting.
func (s *Server) Close() error { return s.ln.Close() }

// Serve accepts connections until the listener closes.
//
// A stage holds one long-lived connection from its upstream neighbour rather
// than reconnecting per token: at 2 KiB a decode hop is far smaller than a TCP
// handshake, so connection setup would dominate the thing being measured.
func (s *Server) Serve() error {
	for {
		conn, err := s.ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go s.handle(conn)
	}
}

func (s *Server) handle(conn net.Conn) {
	defer conn.Close()

	if tcp, ok := conn.(*net.TCPConn); ok {
		// Nagle would hold a small write waiting for company that never
		// arrives in a strict request/response exchange.
		_ = tcp.SetNoDelay(true)
	}

	slog.Debug("shard connection opened", "blocks", s.shard.Blocks(), "peer", conn.RemoteAddr())

	for {
		var req wire.Request
		payload, err := wire.ReadFrame(conn, &req)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("shard connection closed", "blocks", s.shard.Blocks(), "error", err)
			}
			return
		}

		resp, out, err := s.serve(req, payload)
		if err != nil {
			// The failure happened after the request was accepted, so it
			// travels in the response rather than as a transport error.
			resp = wire.Response{Request: req.Request, Error: err.Error()}
			out = nil
		}

		if err := wire.WriteFrame(conn, resp, out); err != nil {
			slog.Debug("shard reply failed", "blocks", s.shard.Blocks(), "error", err)
			return
		}
	}
}

// serve runs one request and measures how long this stage spent on it.
//
// The duration is what the sender subtracts from its own round trip to get time
// genuinely in flight. It is measured here, on this machine's monotonic clock,
// precisely so that no one has to compare clocks across machines.
func (s *Server) serve(req wire.Request, payload []byte) (wire.Response, []byte, error) {
	start := time.Now()

	switch req.Phase {
	case wire.PhaseAlign:
		// An alignment request carries no activation. It reports where this
		// stage's caches rest and, when the coordinator says so, resets them.
		if len(payload) == 0 && req.Token < 0 {
			if err := s.shard.Reset(); err != nil {
				return wire.Response{}, nil, err
			}
		}
		return wire.Response{
			Request:     req.Request,
			Duration:    time.Since(start),
			CacheOffset: s.shard.Offset(),
		}, nil, nil
	}

	dtype, shape, err := describe(req, payload)
	if err != nil {
		return wire.Response{}, nil, err
	}

	outDType, outShape, out, err := s.shard.ForwardBytes(dtype, shape, payload, req.SeqOffsets, req.SeqQueryLens)
	if err != nil {
		return wire.Response{}, nil, err
	}

	return wire.Response{
		Request:     req.Request,
		DType:       outDType.String(),
		Shape:       outShape,
		Duration:    time.Since(start),
		CacheOffset: s.shard.Offset(),
	}, out, nil
}

// describe validates a frame's header against the payload that arrived with
// it, and returns the dtype and shape to rebuild from.
//
// The check matters because a header and payload built from different tensors
// would otherwise reinterpret as a well-formed array of wrong values — a fault
// that surfaces as bad output rather than an error. No MLX array is built here:
// that has to happen on the shard's thread.
func describe(req wire.Request, payload []byte) (mlx.DType, []int, error) {
	if len(req.SeqOffsets) == 0 {
		return 0, nil, errors.New("agent: request carries no sequence offsets")
	}

	var dtype mlx.DType
	if err := dtype.UnmarshalJSON([]byte(`"` + req.DType + `"`)); err != nil {
		return 0, nil, fmt.Errorf("agent: unknown dtype %q", req.DType)
	}

	want, ok := wire.PayloadSize(req.Shape, dtype.ItemSize())
	if !ok {
		return 0, nil, fmt.Errorf("agent: cannot size a %v payload of shape %v", dtype, req.Shape)
	}
	if want != len(payload) {
		return 0, nil, fmt.Errorf("agent: shape %v of %v needs %d bytes, got %d", req.Shape, dtype, want, len(payload))
	}

	return dtype, req.Shape, nil
}
