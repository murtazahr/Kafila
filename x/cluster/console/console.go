// Package console serves the operations dashboard.
//
// Everything the page needs to run ships inside the binary: the HTML, Leaflet,
// and a coastline outline. Only map tiles come from the network, and the page
// falls back to the embedded outline when they do not arrive. That division is
// deliberate — the console is served by a laptop that may be presenting from a
// room with bad wifi, and the failure that matters is the one that takes the
// centre of the screen with it.
//
// It reads two things from the head: the topology, once, and the span stream,
// continuously. Both come from the same source the benchmark reads. A dashboard
// fed by its own separately-derived numbers would eventually disagree with the
// measured ones, and the disagreement would stay invisible until someone chased
// a discrepancy that had been on screen for weeks.
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

// Leaflet 1.9.4, BSD-2-Clause, vendored rather than loaded from a CDN.
//
// The map plots in latitude and longitude and zooms into real tiles, which a
// baked outline cannot do: Natural Earth 110m has no borders and no place
// names, so zooming into it reveals nothing. Leaflet does the projection, so
// the page never converts a coordinate itself.
//
//go:embed lib/leaflet.js
var leafletJS []byte

//go:embed lib/leaflet.css
var leafletCSS []byte

// Natural Earth 110m land outline, public domain, simplified in degrees by
// scripts/build_worldmap.py. It carries latitude and longitude, not pixels:
// Leaflet projects it like anything else.
//
// It is the fallback basemap. When tiles fail to load the page draws this
// instead, so a dead network costs detail rather than the whole map.
//
//go:embed land.geojson
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

	asset := func(contentType string, body []byte) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", contentType)
			w.Header().Set("Cache-Control", "public, max-age=86400")
			_, _ = w.Write(body)
		}
	}

	mux.HandleFunc("GET /assets/land.geojson", asset("application/geo+json", land))
	mux.HandleFunc("GET /assets/leaflet.js", asset("text/javascript; charset=utf-8", leafletJS))
	mux.HandleFunc("GET /assets/leaflet.css", asset("text/css; charset=utf-8", leafletCSS))

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
