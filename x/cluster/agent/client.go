package agent

import (
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Stage is a handle to a shard running elsewhere.
//
// It holds one connection for its lifetime. A decode hop carries 2 KiB, which
// is far less than a TCP handshake costs, so reconnecting per token would make
// connection setup the dominant term in exactly the measurement this exists to
// take.
//
// Nothing here treats a loopback peer differently from a remote one. That is
// deliberate: if the local case took a shortcut, the demo would exercise a code
// path the cluster never uses, and moving to real nodes would be a change of
// code rather than a change of address.
type Stage struct {
	Name    string
	Address string
	Blocks  string

	mu   sync.Mutex
	conn net.Conn
	rec  *trace.Recorder
}

// Dial connects to a shard server.
func Dial(name, address, blocks string, timeout time.Duration) (*Stage, error) {
	conn, err := net.DialTimeout("tcp", address, timeout)
	if err != nil {
		return nil, fmt.Errorf("agent: dial %s (%s): %w", name, address, err)
	}
	if tcp, ok := conn.(*net.TCPConn); ok {
		_ = tcp.SetNoDelay(true)
	}
	return &Stage{Name: name, Address: address, Blocks: blocks, conn: conn}, nil
}

// Trace attaches a recorder. Hops are recorded against it, measured entirely on
// this side's clock.
func (s *Stage) Trace(rec *trace.Recorder) { s.rec = rec }

// Close releases the connection.
func (s *Stage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil {
		return nil
	}
	err := s.conn.Close()
	s.conn = nil
	return err
}

// Forward sends a hidden state to the stage and returns what it produced.
//
// The round trip is timed here, and the stage reports how long it spent on its
// own clock. Subtracting one from the other gives time genuinely in flight
// without either side needing to trust the other's clock.
func (s *Stage) Forward(b *batch.Batch, hidden *mlx.Array, phase wire.Phase, token, chunk int) (*mlx.Array, error) {
	payload, err := hidden.Bytes()
	if err != nil {
		return nil, fmt.Errorf("agent: serialize hidden state: %w", err)
	}

	req := wire.Request{
		Phase:        phase,
		Token:        token,
		Chunk:        chunk,
		DType:        hidden.DType().String(),
		Shape:        hidden.Dims(),
		SeqOffsets:   b.SeqOffsets,
		SeqQueryLens: b.SeqQueryLens,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		return nil, fmt.Errorf("agent: stage %s is closed", s.Name)
	}

	start := time.Now()

	if err := wire.WriteFrame(s.conn, req, payload); err != nil {
		return nil, fmt.Errorf("agent: send to %s: %w", s.Name, err)
	}

	var resp wire.Response
	out, err := wire.ReadFrame(s.conn, &resp)
	if err != nil {
		return nil, fmt.Errorf("agent: receive from %s: %w", s.Name, err)
	}

	rtt := time.Since(start)

	if resp.Error != "" {
		return nil, fmt.Errorf("agent: stage %s: %s", s.Name, resp.Error)
	}

	s.rec.RecordHop(trace.Hop{
		To: s.Name, Phase: trace.Phase(phase), Token: token, Chunk: chunk,
		RoundTrip: rtt, RemoteDuration: resp.Duration,
		// Loopback and a switched LAN are close enough to symmetric to be
		// worth estimating from; the flag is what makes that assumption
		// visible rather than silent.
		Symmetric: true,
		Bytes:     int64(len(payload)),
	})

	var dtype mlx.DType
	if err := dtype.UnmarshalJSON([]byte(`"` + resp.DType + `"`)); err != nil {
		return nil, fmt.Errorf("agent: stage %s returned unknown dtype %q", s.Name, resp.DType)
	}

	want, ok := wire.PayloadSize(resp.Shape, dtype.ItemSize())
	if !ok || want != len(out) {
		return nil, fmt.Errorf("agent: stage %s returned %d bytes for shape %v of %v", s.Name, len(out), resp.Shape, dtype)
	}

	return mlx.FromBytes(out, dtype, resp.Shape...)
}

// Align asks the stage where its caches rest, and resets them when reset is
// set.
//
// Every stage must agree on the token offset, because the offset drives RoPE
// and mask construction; a divergence produces plausible-looking wrong output
// rather than an error. Resetting outright rather than rewinding is the
// conservative choice: not every cache kind can rewind, so reprocessing is the
// only fallback that always works.
func (s *Stage) Align(reset bool) (offset int, err error) {
	req := wire.Request{Phase: wire.PhaseAlign, Token: 0, Chunk: -1}
	if reset {
		req.Token = -1
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn == nil {
		return 0, fmt.Errorf("agent: stage %s is closed", s.Name)
	}

	start := time.Now()
	if err := wire.WriteFrame(s.conn, req, nil); err != nil {
		return 0, fmt.Errorf("agent: align %s: %w", s.Name, err)
	}

	var resp wire.Response
	if _, err := wire.ReadFrame(s.conn, &resp); err != nil {
		return 0, fmt.Errorf("agent: align %s: %w", s.Name, err)
	}

	s.rec.RecordHop(trace.Hop{
		To: s.Name, Phase: trace.PhaseAlign, Token: -1, Chunk: -1,
		RoundTrip: time.Since(start), RemoteDuration: resp.Duration,
		Symmetric: true,
	})

	if resp.Error != "" {
		return 0, fmt.Errorf("agent: stage %s: %s", s.Name, resp.Error)
	}
	return resp.CacheOffset, nil
}
