package agent

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"time"

	"github.com/ollama/ollama/x/cluster/wire"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// The pipeline is a ring. A frame enters at the head, passes through each node
// in turn, and comes back to the head having been round once.
//
// The obvious alternative is a star, where the head sends to each node and
// waits for the answer before sending to the next. It is easier to write and
// worse in two ways that matter. It moves the hidden state four times per
// forward instead of three, which on a 4 MiB prefill chunk is 16 MiB against
// 12. And it puts the head in the path of every hop, so the head can never be
// doing anything else — which forecloses the overlap that makes prefill worth
// splitting at all.
//
// The cost is that no single machine observes a link end to end, so per-link
// timing is no longer available. What is available is better: the head times
// the whole circuit on its own clock, every node reports its own compute on
// its own clock, and the difference is the time genuinely in flight. No
// comparison between clocks anywhere.

// Link sends frames to the next node in the ring.
//
// It holds one connection for its lifetime. A decode frame carries 2 KiB, far
// less than a TCP handshake costs, so reconnecting per token would make
// connection setup the dominant term in the thing being measured.
type Link struct {
	Name    string
	Address string

	mu   sync.Mutex
	conn net.Conn

	// simulated is delay injected before sending, standing in for a link this
	// deployment does not physically have. It is reported separately from
	// anything measured, so a demonstration can never be mistaken for a
	// result.
	simulated time.Duration
}

// DialLink connects to the next node, retrying until the deadline. A ring
// starts as several processes at once and each one loads hundreds of megabytes
// before it listens, so the first attempt routinely arrives too early.
func DialLink(name, address string, timeout time.Duration) (*Link, error) {
	deadline := time.Now().Add(timeout)

	for {
		conn, err := net.DialTimeout("tcp", address, 2*time.Second)
		if err == nil {
			if tcp, ok := conn.(*net.TCPConn); ok {
				_ = tcp.SetNoDelay(true)
			}
			return &Link{Name: name, Address: address, conn: conn}, nil
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("agent: %s at %s did not come up within %s: %w", name, address, timeout, err)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

// Simulate sets an artificial one-way delay on this link.
func (l *Link) Simulate(d time.Duration) { l.simulated = d }

// Simulated reports the injected delay.
func (l *Link) Simulated() time.Duration { return l.simulated }

// Send forwards a frame. The injected delay, if any, is applied here so it sits
// on the wire rather than inside a node's reported compute.
func (l *Link) Send(req wire.Request, payload []byte) error {
	if l.simulated > 0 {
		time.Sleep(l.simulated)
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if l.conn == nil {
		return fmt.Errorf("agent: link to %s is closed", l.Name)
	}
	if err := wire.WriteFrame(l.conn, req, payload); err != nil {
		return fmt.Errorf("agent: send to %s: %w", l.Name, err)
	}
	return nil
}

// Close releases the connection.
func (l *Link) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.conn == nil {
		return nil
	}
	err := l.conn.Close()
	l.conn = nil
	return err
}

// RingNode is a non-head node: it accepts frames from its predecessor, runs its
// blocks, and forwards to its successor.
type RingNode struct {
	shard *Shard
	next  *Link
	ln    net.Listener
}

// ServeRing loads a shard, connects to the next node, and listens for frames.
func ServeRing(spec NodeSpec, dialTimeout time.Duration) (*RingNode, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}

	thread, err := StartThread("shard" + spec.Blocks.String())
	if err != nil {
		return nil, err
	}

	s, err := Load(thread, Config{
		ModelName:  spec.Model,
		Assignment: spec.Assignment(),
		Model:      spec.ModelSpec(),
	})
	if err != nil {
		return nil, err
	}

	ln, err := net.Listen("tcp", spec.Listen)
	if err != nil {
		return nil, fmt.Errorf("agent: listen on %s: %w", spec.Listen, err)
	}

	// Listen before dialling: the predecessor may already be waiting, and a
	// ring where everyone dials first and listens second cannot close.
	next, err := DialLink("next", spec.Next, dialTimeout)
	if err != nil {
		ln.Close()
		return nil, err
	}
	next.Simulate(spec.SimulatedLatency)

	slog.Info("ring node ready",
		"blocks", spec.Blocks, "role", spec.Assignment().Role,
		"listen", ln.Addr(), "next", spec.Next, "simulated", spec.SimulatedLatency)

	return &RingNode{shard: s, next: next, ln: ln}, nil
}

// Addr is the address this node accepts on.
func (n *RingNode) Addr() string { return n.ln.Addr().String() }

// Close stops the node.
func (n *RingNode) Close() error {
	err := n.ln.Close()
	if n.next != nil {
		_ = n.next.Close()
	}
	return err
}

// Serve handles frames until the listener closes.
func (n *RingNode) Serve() error {
	for {
		conn, err := n.ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go n.handle(conn)
	}
}

func (n *RingNode) handle(conn net.Conn) {
	defer conn.Close()

	if tcp, ok := conn.(*net.TCPConn); ok {
		_ = tcp.SetNoDelay(true)
	}

	for {
		var req wire.Request
		payload, err := wire.ReadFrame(conn, &req)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("ring node connection closed", "blocks", n.shard.Blocks(), "error", err)
			}
			return
		}

		out, err := n.step(&req, payload)
		if err != nil {
			// There is no reply path in a ring, so a failure has to travel
			// forward like everything else. The head is the only node that can
			// surface it to a caller.
			slog.Error("ring node failed", "blocks", n.shard.Blocks(), "error", err)
			req.Reports = append(req.Reports, wire.StageReport{
				Node: n.shard.Blocks().String(), CacheOffset: -1,
			})
			out = nil
		}

		if err := n.next.Send(req, out); err != nil {
			slog.Error("ring node could not forward", "blocks", n.shard.Blocks(), "error", err)
			return
		}
	}
}

// step runs this node's part and appends its report to the frame.
func (n *RingNode) step(req *wire.Request, payload []byte) ([]byte, error) {
	if req.Phase == wire.PhaseAlign {
		if req.Reset {
			if err := n.shard.Reset(); err != nil {
				return nil, err
			}
		}
		req.Reports = append(req.Reports, wire.StageReport{
			Node:        n.shard.Blocks().String(),
			CacheOffset: n.shard.Offset(),
			Simulated:   n.next.Simulated(),
		})
		return nil, nil
	}

	dtype, shape, err := describe(*req, payload)
	if err != nil {
		return nil, err
	}

	started := time.Now()
	outDType, outShape, out, err := n.shard.ForwardBytes(dtype, shape, payload, req.SeqOffsets, req.SeqQueryLens)
	if err != nil {
		return nil, err
	}

	req.DType = outDType.String()
	req.Shape = outShape
	req.Reports = append(req.Reports, wire.StageReport{
		Node:        n.shard.Blocks().String(),
		Compute:     time.Since(started),
		CacheOffset: n.shard.Offset(),
		Simulated:   n.next.Simulated(),
	})

	return out, nil
}

// RingReturn is the head's end of the circuit: the listener the last node
// delivers to.
//
// The head blocks on Await while a frame is going round. Requests are processed
// one at a time, so a frame arriving with an unexpected sequence number means
// the ring has lost its place — which would otherwise surface as quietly wrong
// output rather than as an error.
type RingReturn struct {
	ln net.Listener

	mu      sync.Mutex
	waiting map[uint64]chan returned
}

type returned struct {
	req     wire.Request
	payload []byte
	err     error
}

// ListenReturn opens the head's return listener.
func ListenReturn(address string) (*RingReturn, error) {
	ln, err := net.Listen("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("agent: listen for ring returns on %s: %w", address, err)
	}
	return &RingReturn{ln: ln, waiting: map[uint64]chan returned{}}, nil
}

// Addr is the address the last node delivers to.
func (r *RingReturn) Addr() string { return r.ln.Addr().String() }

// Close stops accepting returns.
func (r *RingReturn) Close() error { return r.ln.Close() }

// Serve accepts the tail's connection and dispatches frames to whoever is
// waiting for them.
func (r *RingReturn) Serve() error {
	for {
		conn, err := r.ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go r.handle(conn)
	}
}

func (r *RingReturn) handle(conn net.Conn) {
	defer conn.Close()

	if tcp, ok := conn.(*net.TCPConn); ok {
		_ = tcp.SetNoDelay(true)
	}

	for {
		var req wire.Request
		payload, err := wire.ReadFrame(conn, &req)
		if err != nil {
			return
		}

		r.mu.Lock()
		ch, ok := r.waiting[req.Seq]
		delete(r.waiting, req.Seq)
		r.mu.Unlock()

		if !ok {
			slog.Warn("ring: a frame came back that nothing was waiting for", "seq", req.Seq)
			continue
		}
		ch <- returned{req: req, payload: payload}
	}
}

// Expect registers interest in a circuit before it is dispatched, so a fast
// ring cannot deliver before the head is listening for it.
func (r *RingReturn) Expect(seq uint64) <-chan returned {
	ch := make(chan returned, 1)
	r.mu.Lock()
	r.waiting[seq] = ch
	r.mu.Unlock()
	return ch
}

// Forget drops a registration whose circuit failed or timed out.
func (r *RingReturn) Forget(seq uint64) {
	r.mu.Lock()
	delete(r.waiting, seq)
	r.mu.Unlock()
}

// rebuild turns a returned frame back into an array on the caller's thread.
func rebuild(req wire.Request, payload []byte) (*mlx.Array, error) {
	var dtype mlx.DType
	if err := dtype.UnmarshalJSON([]byte(`"` + req.DType + `"`)); err != nil {
		return nil, fmt.Errorf("agent: ring returned unknown dtype %q", req.DType)
	}
	want, ok := wire.PayloadSize(req.Shape, dtype.ItemSize())
	if !ok || want != len(payload) {
		return nil, fmt.Errorf("agent: ring returned %d bytes for shape %v of %v", len(payload), req.Shape, dtype)
	}
	return mlx.FromBytes(payload, dtype, req.Shape...)
}
