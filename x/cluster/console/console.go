// Package console serves the operations dashboard.
//
// The page is one self-contained HTML file with no external assets, because it
// is served by a laptop that may be presenting from a room with no network. It
// reads two things: the topology, once, and the span stream, continuously.
//
// Both come from the same source the benchmark reads. A dashboard fed by its
// own separately-derived numbers would eventually disagree with the measured
// ones, and the disagreement would stay invisible until someone chased a
// discrepancy that had been on screen for weeks.
package console

import (
	_ "embed"
	"fmt"
	"net/http"
	"time"

	"github.com/ollama/ollama/x/cluster/trace"
)

//go:embed console.html
var page []byte

// Natural Earth 110m land outline, already projected into the page's viewBox
// and simplified. Kept as its own asset rather than inlined in the HTML: it is
// data, and mixing 30 KB of coordinates into a presentation file makes both
// harder to read.
//
// It ships in the binary rather than coming from a tile provider because the
// console is served by a machine that may be presenting from a room with no
// network. A map that silently fails to load would take the centre of the
// screen with it, and no CDN is worth that.
//
//go:embed land.path
var land []byte

// Register adds the dashboard and its event stream to a mux.
//
// The topology handler is supplied by the caller because only the head knows
// the pipeline's shape; everything else here is presentation.
func Register(mux *http.ServeMux, b *trace.Broadcaster, topology http.HandlerFunc) {
	// "{$}" anchors to the exact root. A bare "GET /" is a wildcard that
	// conflicts with the runner's method-less routes, and Go rejects the
	// ambiguity at registration rather than resolving it silently.
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(page)
	})

	mux.HandleFunc("GET /assets/land.path", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(land)
	})

	mux.HandleFunc("GET /v1/topology", topology)
	mux.HandleFunc("GET /v1/events", eventStream(b))
}

// eventStream serves the span stream as Server-Sent Events.
//
// Events are forwarded as they are recorded, which during decode is a few
// hundred a second. The handler flushes each one rather than letting the writer
// buffer: a dashboard whose purpose is showing a token move through the
// pipeline is worthless if the movement arrives in batches.
func eventStream(b *trace.Broadcaster) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		events, release := b.Subscribe()
		defer release()

		// An opening comment makes the connection usable immediately rather
		// than when the first token happens to be generated.
		fmt.Fprint(w, ": connected\n\n")
		flusher.Flush()

		// Proxies and browsers drop a stream that goes quiet. An idle cluster
		// is the normal state between requests, so it has to say so.
		keepalive := time.NewTicker(15 * time.Second)
		defer keepalive.Stop()

		for {
			select {
			case <-r.Context().Done():
				return

			case line, open := <-events:
				if !open {
					return
				}
				// SSE frames are newline-delimited, and a span is one NDJSON
				// line, so it needs no escaping beyond trimming its own.
				fmt.Fprintf(w, "data: %s\n\n", trimNewline(line))
				flusher.Flush()

			case <-keepalive.C:
				fmt.Fprint(w, ": keepalive\n\n")
				flusher.Flush()
			}
		}
	}
}

func trimNewline(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}
