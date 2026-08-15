package mlxrunner

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/cluster/agent"
	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// ExecuteShard runs one node of a split model.
//
// Every node runs this same entrypoint and differs only in flags. A node that
// is not the head loads its blocks and serves them; the head loads its blocks,
// connects to the others, and presents the whole pipeline through the ordinary
// runner HTTP interface, so nothing above it can tell the model is split.
//
// One shard per process is not a demo convenience. Two shards evaluating
// concurrently in one process deadlock inside MLX, which keeps global state not
// built for several independent models at once.
func ExecuteShard(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	var (
		modelName string
		blocksArg string
		totalArg  int
		tied      bool
		isHead    bool
		isTail    bool
		listen    string
		stagesArg string
		port      int
		tracePath string
	)

	fs := flag.NewFlagSet("shardnode", flag.ExitOnError)
	fs.StringVar(&modelName, "model", "", "safetensors model name")
	fs.StringVar(&blocksArg, "blocks", "", "block range this node owns, as start:end")
	fs.IntVar(&totalArg, "total-blocks", 0, "the model's full block count")
	fs.BoolVar(&tied, "tied-embeddings", false, "the checkpoint ties its output projection to the embedding")
	fs.BoolVar(&isHead, "head", false, "own the embedding, output projection and sampler")
	fs.BoolVar(&isTail, "tail", false, "own the final norm")
	fs.StringVar(&listen, "listen", "", "address to serve this shard on (non-head nodes)")
	fs.StringVar(&stagesArg, "stages", "", "comma-separated addresses of downstream nodes, in order (head only)")
	fs.IntVar(&port, "port", 0, "HTTP port for the runner interface (head only)")
	fs.StringVar(&tracePath, "trace", "", "write the NDJSON span stream here")
	if err := fs.Parse(args); err != nil {
		return err
	}

	blocks, err := agent.ParseRange(blocksArg)
	if err != nil {
		return err
	}

	spec := agent.NodeSpec{
		Model:          modelName,
		Blocks:         blocks,
		Head:           isHead,
		Tail:           isTail,
		Listen:         listen,
		TotalBlocks:    totalArg,
		TiedEmbeddings: tied,
	}
	if stagesArg != "" {
		spec.Stages = strings.Split(stagesArg, ",")
	}
	// Each downstream address may carry the block range that node owns, as
	// addr=start:end. It changes nothing about how the pipeline runs and
	// everything about whether the topology can be described.
	for i, entry := range spec.Stages {
		addr, blocksPart, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		r, err := agent.ParseRange(blocksPart)
		if err != nil {
			return err
		}
		spec.Stages[i] = addr
		spec.StageBlocks = append(spec.StageBlocks, r)
	}
	if err := spec.Validate(); err != nil {
		return err
	}

	if !spec.Head {
		return serveShardNode(spec)
	}
	return runHeadNode(spec, port, tracePath)
}

// serveShardNode is what a non-head process runs: load the blocks, serve them,
// and stay up until it is killed.
func serveShardNode(spec agent.NodeSpec) error {
	srv, err := agent.ServeNode(spec)
	if err != nil {
		return err
	}
	defer srv.Close()

	return srv.Serve()
}

// runHeadNode loads the head's blocks, connects to the stages downstream, and
// serves the composed pipeline through the runner's own HTTP interface.
func runHeadNode(spec agent.NodeSpec, port int, tracePath string) error {
	worker, err := agent.StartThread("shardhead")
	if err != nil {
		return err
	}
	defer worker.Stop(context.Background(), func() {
		mlx.Sweep()
		mlx.ClearCache()
	})

	// The head takes no thread of its own for its shard: the runner already
	// drives it from inside this worker, and hopping onto a worker from within
	// that worker waits on itself.
	var head *agent.Shard
	if err := worker.Do(context.Background(), func() error {
		var err error
		head, err = agent.Load(nil, agent.Config{
			ModelName:  spec.Model,
			Assignment: spec.Assignment(),
			Model:      spec.ModelSpec(),
		})
		return err
	}); err != nil {
		return fmt.Errorf("load head shard: %w", err)
	}

	stages, err := agent.DialStages(spec.Stages, spec.StageBlocks, 120*time.Second)
	if err != nil {
		return err
	}
	defer func() {
		for _, s := range stages {
			_ = s.Close()
		}
	}()

	pipeline, err := agent.NewPipelineModel(head, stages)
	if err != nil {
		return err
	}

	if tracePath != "" {
		f, err := os.OpenFile(tracePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("open trace: %w", err)
		}
		defer f.Close()
		pipeline.Trace(trace.NewRecorder(f, "pipeline", "head", spec.Blocks))
	}

	// Every stage must start from the same offset, or their positions have
	// already drifted before a single token is processed.
	if err := pipeline.Align(); err != nil {
		return fmt.Errorf("align stages: %w", err)
	}

	runner := Runner{
		Requests:  make(chan Request),
		mlxThread: worker,
	}
	if err := worker.Do(context.Background(), func() error {
		return runner.Attach(pipeline)
	}); err != nil {
		return err
	}

	slog.Info("pipeline head ready",
		"blocks", spec.Blocks, "stages", len(stages),
		"context", runner.contextLength, "port", port)

	// The head serves the same endpoints an unsplit runner does, so nothing
	// above it can tell the model is split, plus one the dashboard reads.
	mux := runner.newMux(func() uint64 {
		return uint64(mlx.ActiveMemory() + mlx.CacheMemory())
	})
	mux.HandleFunc("GET /v1/topology", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(describeTopology(pipeline, spec)); err != nil {
			slog.Error("encode topology", "error", err)
		}
	})

	return runner.Run("127.0.0.1", strconv.Itoa(port), mux)
}

// describeTopology reports the shape of the pipeline: which node holds which
// blocks, and where each one is. It is what a dashboard renders and what a
// future coordinator would use to check a plan against reality.
func describeTopology(pipeline *agent.PipelineModel, spec agent.NodeSpec) map[string]any {
	nodes := []map[string]any{{
		"index":   0,
		"name":    "head",
		"blocks":  spec.Blocks.String(),
		"start":   spec.Blocks.Start,
		"end":     spec.Blocks.End,
		"role":    spec.Assignment().Role.String(),
		"address": "local",
	}}

	for i, s := range pipeline.Stages() {
		nodes = append(nodes, map[string]any{
			"index":   i + 1,
			"name":    s.Name,
			"blocks":  s.Blocks,
			"role":    stageRole(i, len(pipeline.Stages())),
			"address": s.Address,
		})
	}

	return map[string]any{
		"model":       spec.Model,
		"blocks":      spec.TotalBlocks,
		"tied":        spec.TiedEmbeddings,
		"stage_count": len(nodes),
		"nodes":       nodes,
	}
}

func stageRole(i, n int) string {
	if i == n-1 {
		return shard.Tail.String()
	}
	return "middle"
}
