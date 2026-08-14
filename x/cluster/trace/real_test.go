package trace

import (
	"os"
	"testing"
)

// TestParseRealTrace reads a span stream produced by a running server, rather
// than one this package wrote itself.
//
// The round trip through the recorder is already covered; what this checks is
// that a real request produces a trace with the shape the analysis assumes —
// phases present, tokens indexed, and the residual small enough that the
// instrumentation is actually describing where the time went.
//
//	OLLAMA_EXPERIMENT=cluster OLLAMA_CLUSTER_TRACE=/tmp/t.ndjson ollama serve
//	OLLAMA_TEST_TRACE=/tmp/t.ndjson go test ./x/cluster/trace/ -run RealTrace -v
func TestParseRealTrace(t *testing.T) {
	path := os.Getenv("OLLAMA_TEST_TRACE")
	if path == "" {
		t.Skip("set OLLAMA_TEST_TRACE to a span stream from a running server")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open trace: %v", err)
	}
	defer f.Close()

	sums, err := Parse(f)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(sums) == 0 {
		t.Fatal("trace contains no requests")
	}

	generations := 0
	for _, sum := range sums {
		for _, n := range sum.Nodes {
			t.Logf("%-8s %-16s %3d spans | elapsed %10s | accounted %10s | residual %10s",
				sum.Request, n.Node, n.Spans, n.Elapsed, n.Accounted, n.Residual)
			for _, p := range []Phase{PhaseLoad, PhasePrefill, PhaseDecode} {
				if d := n.ByPhase[p]; d > 0 {
					t.Logf("           %-10s %s", p, d)
				}
			}

			if n.Overlapped {
				t.Errorf("%s/%s: spans overlapped, so the residual is meaningless", sum.Request, n.Node)
				continue
			}

			// A generation is a request that produced tokens. Load-only
			// requests have nothing to say about coverage.
			if n.ByPhase[PhaseDecode] == 0 {
				continue
			}
			generations++

			// The wrapper spans should account for nearly all of the request:
			// they are contiguous by construction, so a large residual means
			// something is being missed rather than merely unmeasured.
			if n.Elapsed > 0 {
				share := float64(n.Residual) / float64(n.Elapsed)
				if share > 0.05 {
					t.Errorf("%s/%s: %.1f%% of elapsed time is unattributed",
						sum.Request, n.Node, 100*share)
				}
			}
		}

		if sum.InconsistentHops > 0 {
			t.Errorf("%s: %d hops whose sender and receiver measurements disagree",
				sum.Request, sum.InconsistentHops)
		}
	}

	if generations == 0 {
		t.Fatal("no request in the trace produced tokens")
	}
	t.Logf("%d request(s), %d generation(s)", len(sums), generations)
}
