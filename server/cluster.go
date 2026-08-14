package server

import (
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/cluster"
	"github.com/ollama/ollama/x/mlxrunner"
)

// clusterEnabled reports whether inference should be routed through the
// distributed pipeline. Off by default: with no other nodes registered the
// pipeline is a single-stage passthrough, so the only thing enabling it changes
// is that the control plane and the trace plane are exercised.
func clusterEnabled() bool { return experimentEnabled("cluster") }

// clusterTracePath is where the NDJSON span stream is written. Tracing is off
// unless a path is given, so enabling the cluster never creates files nobody
// asked for.
func clusterTracePath() string { return os.Getenv("OLLAMA_CLUSTER_TRACE") }

// newClusterPipeline wraps a local MLX runner in a single-node cluster
// pipeline.
//
// This is Stage 1's degenerate case and it distributes nothing. Its value is
// that planning, tensor selection, tracing and the llm.LlamaServer façade all
// run on the real request path while inference itself remains byte-identical to
// the unsharded runner — so anything that breaks here is the new machinery
// rather than the split.
//
// A failure to build the pipeline returns the unwrapped runner rather than
// failing the load. Losing the control plane should degrade to ordinary
// inference, not deny service.
func newClusterPipeline(modelName string, local llm.LlamaServer) llm.LlamaServer {
	m, err := mlxrunner.Inspect(modelName)
	if err != nil {
		slog.Warn("cluster: could not inspect model; serving unwrapped", "model", modelName, "error", err)
		return local
	}

	registry := cluster.NewRegistry()
	if err := registry.Add(cluster.Node{Name: clusterLocalNodeName(), Library: "Metal"}); err != nil {
		slog.Warn("cluster: could not register the local node; serving unwrapped", "error", err)
		return local
	}

	plan, err := cluster.Build(registry, m)
	if err != nil {
		slog.Warn("cluster: could not plan; serving unwrapped", "model", modelName, "error", err)
		return local
	}

	traceOut, err := openClusterTrace()
	if err != nil {
		slog.Warn("cluster: could not open the trace file; continuing untraced", "error", err)
	}

	p, err := cluster.NewPipeline(plan, []llm.LlamaServer{local}, traceOut)
	if err != nil {
		slog.Warn("cluster: could not build the pipeline; serving unwrapped", "error", err)
		if c, ok := traceOut.(io.Closer); ok && traceOut != nil {
			_ = c.Close()
		}
		return local
	}

	slog.Info("cluster: serving through the distributed pipeline",
		"model", modelName, "plan", plan.String(), "traced", traceOut != nil)

	return p
}

// clusterLocalNodeName identifies this machine in plans and traces. The
// hostname is stable enough for a single-node cluster and meaningful once there
// are others.
func clusterLocalNodeName() string {
	if h, err := os.Hostname(); err == nil && h != "" {
		return h
	}
	return "local"
}

// openClusterTrace opens the span stream, appending so that runs accumulate
// rather than the last one silently erasing the evidence from the previous.
// Returns a nil writer, and no error, when tracing is not configured.
func openClusterTrace() (io.Writer, error) {
	path := clusterTracePath()
	if path == "" {
		return nil, nil
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	return f, nil
}
