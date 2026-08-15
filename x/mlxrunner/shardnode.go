package mlxrunner

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/cluster/agent"
	"github.com/ollama/ollama/x/cluster/console"
	"github.com/ollama/ollama/x/cluster/trace"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
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
		describe  bool
		nextAddr  string
		returnAt  string
		simulated time.Duration
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
	fs.BoolVar(&describe, "describe", false, "print the model's block count and tie flag, then exit")
	fs.StringVar(&nextAddr, "next", "", "address of the next node in the ring")
	fs.StringVar(&returnAt, "return-listen", "", "address the last node delivers back to (head only)")
	fs.DurationVar(&simulated, "simulate-latency", 0, "inject a one-way delay on the link to the next node, standing in for distance this deployment does not have")
	if err := fs.Parse(args); err != nil {
		return err
	}

	// A planner needs the block count and the tie flag before it can divide
	// anything, and both come from the manifest rather than being guessed.
	if describe {
		spec, err := model.Inspect(modelName)
		if err != nil {
			return err
		}
		fmt.Printf("%d %t\n", spec.Blocks, spec.TiedEmbeddings)
		return nil
	}

	blocks, err := agent.ParseRange(blocksArg)
	if err != nil {
		return err
	}

	spec := agent.NodeSpec{
		Model:            modelName,
		Blocks:           blocks,
		Head:             isHead,
		Tail:             isTail,
		Listen:           listen,
		TotalBlocks:      totalArg,
		TiedEmbeddings:   tied,
		Next:             nextAddr,
		SimulatedLatency: simulated,
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
	return runHeadNode(spec, port, tracePath, returnAt)
}

// serveShardNode is what a non-head process runs: load the blocks, serve them,
// and stay up until it is killed.
func serveShardNode(spec agent.NodeSpec) error {
	node, err := agent.ServeRing(spec, 3*time.Minute)
	if err != nil {
		return err
	}
	defer node.Close()

	return node.Serve()
}

// runHeadNode loads the head's blocks, connects to the stages downstream, and
// serves the composed pipeline through the runner's own HTTP interface.
func runHeadNode(spec agent.NodeSpec, port int, tracePath, returnAt string) error {
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

	// The head listens for the frame to come back round before it dials
	// onward: the last node connects here, and a ring where everyone dials
	// before listening cannot close.
	ret, err := agent.ListenReturn(returnAt)
	if err != nil {
		return err
	}
	defer ret.Close()
	go func() {
		if err := ret.Serve(); err != nil {
			slog.Error("ring return listener stopped", "error", err)
		}
	}()

	next, err := agent.DialLink("next", spec.Next, 3*time.Minute)
	if err != nil {
		return err
	}
	defer next.Close()
	next.Simulate(spec.SimulatedLatency)

	pipeline, err := agent.NewRingModel(head, next, ret, describeStages(spec))
	if err != nil {
		return err
	}

	// One stream feeds the durable trace and the live console. A dashboard on
	// a separately-derived feed would eventually disagree with the benchmark,
	// and the disagreement would stay invisible.
	events := trace.NewBroadcaster()
	var sink io.Writer = events

	if tracePath != "" {
		f, err := os.OpenFile(tracePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("open trace: %w", err)
		}
		defer f.Close()
		sink = io.MultiWriter(f, events)
	}
	pipeline.Trace(trace.NewRecorder(sink, "pipeline", "head", spec.Blocks))

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

	slog.Info("ring head ready",
		"blocks", spec.Blocks, "nodes", len(spec.Stages)+1,
		"next", spec.Next, "return", ret.Addr(),
		"context", runner.contextLength, "port", port)

	// The head serves the same endpoints an unsplit runner does, so nothing
	// above it can tell the model is split, plus one the dashboard reads.
	mux := runner.newMux(func() uint64 {
		return uint64(mlx.ActiveMemory() + mlx.CacheMemory())
	})
	console.Register(mux, events, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(describeTopology(pipeline, spec)); err != nil {
			slog.Error("encode topology", "error", err)
		}
	})

	slog.Info("console ready", "url", fmt.Sprintf("http://127.0.0.1:%d/", port))

	return runner.Run("127.0.0.1", strconv.Itoa(port), mux)
}

// describeTopology reports the shape of the pipeline: which node holds which
// blocks, and where each one is. It is what a dashboard renders and what a
// future coordinator would use to check a plan against reality.
// describeStages renders the nodes downstream from the launch flags. The head
// does not talk to them directly in a ring, so their details come from the plan
// rather than from a connection.
func describeStages(spec agent.NodeSpec) []agent.StageInfo {
	out := make([]agent.StageInfo, 0, len(spec.Stages))
	for i, addr := range spec.Stages {
		blocks := ""
		if i < len(spec.StageBlocks) {
			blocks = spec.StageBlocks[i].String()
		}
		role := "middle"
		if i == len(spec.Stages)-1 {
			role = shard.Tail.String()
		}
		out = append(out, agent.StageInfo{
			Name: fmt.Sprintf("stage%d", i+1), Blocks: blocks, Address: addr, Role: role,
		})
	}
	return out
}

func describeTopology(pipeline *agent.RingModel, spec agent.NodeSpec) map[string]any {
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
			"role":    s.Role,
			"address": s.Address,
		})
	}

	return map[string]any{
		"topology":    "ring",
		"model":       spec.Model,
		"blocks":      spec.TotalBlocks,
		"tied":        spec.TiedEmbeddings,
		"stage_count": len(nodes),
		"nodes":       nodes,
	}
}
