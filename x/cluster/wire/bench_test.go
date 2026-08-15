package wire

import (
	"bytes"
	"fmt"
	"net"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

// The transport's job is to move an activation between two stages and get an
// answer back. Which protocol carries it matters far less than people expect at
// these sizes, so these two tests measure the link rather than argue about it.
//
// On the responder:
//
//	WIRE_ECHO_LISTEN=:9900 go test ./x/cluster/wire/ -run TestEchoServer -timeout 0
//
// On the sender:
//
//	WIRE_ECHO_DIAL=host:9900 go test ./x/cluster/wire/ -run TestRoundTripLatency -v
//
// Both skip when their variable is unset, so an ordinary test run ignores them.

// TestEchoServer serves frames back to the sender. It is the minimum a stage
// does with the transport — read a frame, reply with one — so a round trip
// against it measures the floor that a real shard can never beat.
func TestEchoServer(t *testing.T) {
	addr := os.Getenv("WIRE_ECHO_LISTEN")
	if addr == "" {
		t.Skip("set WIRE_ECHO_LISTEN to run the responder")
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	t.Logf("echo responder on %s", ln.Addr())

	for {
		conn, err := ln.Accept()
		if err != nil {
			t.Logf("accept: %v", err)
			return
		}

		go func() {
			defer conn.Close()
			if tcp, ok := conn.(*net.TCPConn); ok {
				// Without this, Nagle holds a small write waiting for more
				// data and can add tens of milliseconds to a 2 KiB exchange.
				_ = tcp.SetNoDelay(true)
			}
			for {
				var req Request
				payload, err := ReadFrame(conn, &req)
				if err != nil {
					return
				}
				start := time.Now()
				resp := Response{
					Request:  req.Request,
					DType:    req.DType,
					Shape:    req.Shape,
					Duration: time.Since(start),
				}
				if err := WriteFrame(conn, resp, payload); err != nil {
					return
				}
			}
		}()
	}
}

// TestRoundTripLatency measures a framed exchange at the payload sizes the
// pipeline actually uses, with and without Nagle.
//
// A decode hop carries one hidden state — 2 KiB at hidden 1024 in bf16 — and
// prefill carries a chunk, 4 MiB at 2048 tokens. Those differ by three orders
// of magnitude and are bound by different things: the small one by round-trip
// time and syscalls, the large one by bandwidth.
func TestRoundTripLatency(t *testing.T) {
	addr := os.Getenv("WIRE_ECHO_DIAL")
	if addr == "" {
		t.Skip("set WIRE_ECHO_DIAL to a responder address")
	}

	iterations := 200
	if s := os.Getenv("WIRE_ECHO_ITERS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			iterations = n
		}
	}

	sizes := []struct {
		label string
		shape []int
	}{
		{"decode token   2 KiB", []int{1, 1, 1024}},
		{"8 tokens      16 KiB", []int{1, 8, 1024}},
		{"128 tokens   256 KiB", []int{1, 128, 1024}},
		{"prefill chunk  4 MiB", []int{1, 2048, 1024}},
	}

	for _, noDelay := range []bool{true, false} {
		label := "TCP_NODELAY on"
		if !noDelay {
			label = "TCP_NODELAY off (Nagle)"
		}
		t.Logf("--- %s ---", label)

		for _, sz := range sizes {
			n, ok := PayloadSize(sz.shape, 2)
			if !ok {
				t.Fatalf("bad shape %v", sz.shape)
			}
			stats, err := measureRoundTrip(addr, noDelay, sz.shape, n, iterations)
			if err != nil {
				t.Fatalf("%s: %v", sz.label, err)
			}
			t.Logf("  %-22s min %8s  p50 %8s  p99 %8s   %s",
				sz.label, stats.min, stats.p50, stats.p99, throughput(n, stats.p50))
		}
	}
}

type rttStats struct{ min, p50, p99 time.Duration }

func measureRoundTrip(addr string, noDelay bool, shape []int, payloadSize, iterations int) (rttStats, error) {
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return rttStats{}, err
	}
	defer conn.Close()

	if tcp, ok := conn.(*net.TCPConn); ok {
		if err := tcp.SetNoDelay(noDelay); err != nil {
			return rttStats{}, err
		}
	}

	payload := bytes.Repeat([]byte{0xA5}, payloadSize)
	req := Request{
		Request: "bench", Phase: PhaseDecode, Token: 0, Chunk: -1,
		DType: "BF16", Shape: shape,
		SeqOffsets: []int32{0}, SeqQueryLens: []int32{1},
	}

	// Warm the connection so the first measurement is not paying for slow
	// start or a cold path.
	for range 5 {
		if err := WriteFrame(conn, req, payload); err != nil {
			return rttStats{}, err
		}
		var resp Response
		if _, err := ReadFrame(conn, &resp); err != nil {
			return rttStats{}, err
		}
	}

	samples := make([]time.Duration, 0, iterations)
	for range iterations {
		start := time.Now()
		if err := WriteFrame(conn, req, payload); err != nil {
			return rttStats{}, err
		}
		var resp Response
		if _, err := ReadFrame(conn, &resp); err != nil {
			return rttStats{}, err
		}
		samples = append(samples, time.Since(start))
	}

	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	return rttStats{
		min: samples[0],
		p50: samples[len(samples)/2],
		p99: samples[min(len(samples)*99/100, len(samples)-1)],
	}, nil
}

func throughput(bytes int, d time.Duration) string {
	if d <= 0 {
		return ""
	}
	// Round trip moves the payload twice.
	mbps := float64(bytes) * 2 / d.Seconds() / (1 << 20)
	return fmt.Sprintf("%7.1f MiB/s", mbps)
}
