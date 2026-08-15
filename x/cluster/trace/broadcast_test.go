package trace

import (
	"bytes"
	"encoding/json"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

func TestBroadcastReachesSubscribers(t *testing.T) {
	b := NewBroadcaster()

	one, releaseOne := b.Subscribe()
	two, releaseTwo := b.Subscribe()
	defer releaseOne()
	defer releaseTwo()

	if got := b.Subscribers(); got != 2 {
		t.Fatalf("Subscribers = %d, want 2", got)
	}

	rec := NewRecorder(b, "req-1", "head", shard.Range{Start: 0, End: 10})
	rec.Record(PhaseDecode, KindCompute, time.Millisecond, Token(3))

	for name, ch := range map[string]<-chan []byte{"one": one, "two": two} {
		select {
		case line := <-ch:
			var s Span
			if err := json.Unmarshal(line, &s); err != nil {
				t.Fatalf("%s received unparseable line: %v", name, err)
			}
			if s.Token != 3 || s.Node != "head" {
				t.Errorf("%s received %+v", name, s)
			}
		case <-time.After(2 * time.Second):
			t.Errorf("%s received nothing", name)
		}
	}
}

// The same stream has to feed the durable trace and the live view, or the
// dashboard and the benchmark will eventually disagree about what happened.
func TestBroadcastComposesWithAFile(t *testing.T) {
	var file bytes.Buffer
	b := NewBroadcaster()

	ch, release := b.Subscribe()
	defer release()

	rec := NewRecorder(io.MultiWriter(&file, b), "req-1", "head", shard.Range{})
	rec.Record(PhaseDecode, KindCompute, time.Millisecond, Token(0))

	select {
	case line := <-ch:
		if !bytes.Equal(bytes.TrimSpace(line), bytes.TrimSpace(file.Bytes())) {
			t.Errorf("live and durable streams differ:\nlive %s\nfile %s", line, file.Bytes())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("subscriber received nothing")
	}
}

// Recording must never slow inference down. A subscriber that cannot keep up
// loses events rather than becoming the reason tokens are late.
func TestSlowSubscriberIsDroppedNotBlocking(t *testing.T) {
	b := NewBroadcaster()
	_, release := b.Subscribe()
	defer release()

	rec := NewRecorder(b, "req-1", "head", shard.Range{})

	done := make(chan struct{})
	go func() {
		defer close(done)
		// Far more than the channel buffer, with nothing reading.
		for i := range 5000 {
			rec.Record(PhaseDecode, KindCompute, time.Microsecond, Token(i))
		}
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("writing blocked on a subscriber that was not reading")
	}
}

// A released subscriber must stop receiving, and releasing twice must not
// panic on a closed channel.
func TestReleaseIsIdempotent(t *testing.T) {
	b := NewBroadcaster()
	ch, release := b.Subscribe()

	release()
	release()

	if got := b.Subscribers(); got != 0 {
		t.Errorf("Subscribers = %d after release, want 0", got)
	}
	if _, open := <-ch; open {
		t.Error("channel still open after release")
	}
}

func TestBroadcastIsConcurrencySafe(t *testing.T) {
	b := NewBroadcaster()

	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch, release := b.Subscribe()
			defer release()
			select {
			case <-ch:
			case <-time.After(100 * time.Millisecond):
			}
		}()
	}
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rec := NewRecorder(b, "req", "head", shard.Range{})
			rec.Record(PhaseDecode, KindCompute, time.Microsecond)
		}()
	}
	wg.Wait()

	if got := b.Subscribers(); got != 0 {
		t.Errorf("Subscribers = %d after everyone released, want 0", got)
	}
}

// A nil broadcaster stays usable so the head can be wired unconditionally.
func TestNilBroadcasterIsInert(t *testing.T) {
	var b *Broadcaster
	if n, err := b.Write([]byte("x")); n != 1 || err != nil {
		t.Errorf("nil Write = %d, %v", n, err)
	}
	if got := b.Subscribers(); got != 0 {
		t.Errorf("nil Subscribers = %d", got)
	}
}
