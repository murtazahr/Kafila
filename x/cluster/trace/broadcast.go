package trace

import (
	"sync"
)

// Broadcaster fans the span stream out to live subscribers while it is being
// written.
//
// It is an io.Writer, so it composes with the file the recorder already writes
// to: one stream feeds both the durable trace and anything watching. That
// matters because a dashboard built on a second, separately-derived feed would
// eventually disagree with the benchmark, and the disagreement would be
// invisible until someone chased a discrepancy.
//
// A slow subscriber is dropped rather than allowed to block. Recording must
// never slow inference down, and a viewer that cannot keep up with a decode
// loop is better off missing events than becoming the reason tokens are late.
type Broadcaster struct {
	mu   sync.RWMutex
	subs map[int]chan []byte
	next int
}

// NewBroadcaster returns an empty broadcaster.
func NewBroadcaster() *Broadcaster {
	return &Broadcaster{subs: map[int]chan []byte{}}
}

// Write publishes one NDJSON line. It never blocks and never errors: the
// recorder treats a failed write as a trace problem, and a subscriber falling
// behind is not one.
func (b *Broadcaster) Write(p []byte) (int, error) {
	if b == nil {
		return len(p), nil
	}

	// The recorder reuses its encoding buffer, so a subscriber that reads the
	// slice later would see whatever was written next.
	line := make([]byte, len(p))
	copy(line, p)

	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, ch := range b.subs {
		select {
		case ch <- line:
		default:
			// Full: this subscriber is behind. Drop the event rather than
			// stall the request that produced it.
		}
	}

	return len(p), nil
}

// Subscribe returns a channel of NDJSON lines and a function to release it.
//
// The buffer is deep enough to absorb a burst — a decode loop emits a few
// hundred events a second — without letting an abandoned subscriber grow
// without bound.
func (b *Broadcaster) Subscribe() (<-chan []byte, func()) {
	ch := make(chan []byte, 512)

	b.mu.Lock()
	id := b.next
	b.next++
	b.subs[id] = ch
	b.mu.Unlock()

	return ch, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if existing, ok := b.subs[id]; ok {
			delete(b.subs, id)
			close(existing)
		}
	}
}

// Subscribers reports how many are currently attached.
func (b *Broadcaster) Subscribers() int {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subs)
}
