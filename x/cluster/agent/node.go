package agent

import (
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// NodeSpec describes one process's part in a split.
//
// A node is one shard in one process. That is not a simplification for the
// demo: two shards evaluating concurrently inside a single process deadlock
// inside MLX, which keeps global state that is not built for several
// independent models at once. One shard per process is the shape that works and
// the shape a cluster has anyway.
type NodeSpec struct {
	// Model is the safetensors model every node loads its slice of.
	Model string

	// Blocks is the half-open block range this node owns.
	Blocks shard.Range

	// Head and Tail say which ends of the pipeline this node holds. The head
	// embeds tokens, unembeds and samples; the tail applies the final norm.
	Head bool
	Tail bool

	// Listen is the address a non-head node serves on.
	Listen string

	// Stages are the addresses of the nodes downstream of this one, in
	// pipeline order. Only the head sets these.
	Stages []string

	// StageBlocks are the block ranges those nodes own, in the same order.
	// The head does not need them to run, but reporting the topology without
	// them leaves a dashboard unable to say who holds what.
	StageBlocks []shard.Range

	// TotalBlocks is the model's full block count, used to check the plan
	// covers it.
	TotalBlocks int

	// TiedEmbeddings mirrors the checkpoint's tie flag so a node skips the
	// redundant lm_head.
	TiedEmbeddings bool
}

// Assignment renders the spec as a shard assignment.
func (n NodeSpec) Assignment() shard.Assignment {
	var role shard.Role
	if n.Head {
		role |= shard.Head
	}
	if n.Tail {
		role |= shard.Tail
	}
	return shard.Assignment{Range: n.Blocks, Role: role}
}

// ModelSpec renders the checkpoint facts a shard needs.
func (n NodeSpec) ModelSpec() shard.Model {
	return shard.Model{Blocks: n.TotalBlocks, TiedEmbeddings: n.TiedEmbeddings}
}

// Validate checks the spec is coherent before anything is loaded.
func (n NodeSpec) Validate() error {
	if n.Model == "" {
		return errors.New("agent: node needs a model")
	}
	if n.Blocks.Len() <= 0 {
		return fmt.Errorf("agent: node owns no blocks: %s", n.Blocks)
	}
	if n.TotalBlocks > 0 && n.Blocks.End > n.TotalBlocks {
		return fmt.Errorf("agent: block range %s exceeds the model's %d blocks", n.Blocks, n.TotalBlocks)
	}
	if n.Head && n.Listen != "" && len(n.Stages) == 0 {
		return errors.New("agent: a head with no downstream stages should run unsplit")
	}
	if !n.Head && len(n.Stages) > 0 {
		return errors.New("agent: only the head drives downstream stages")
	}
	if !n.Head && n.Listen == "" {
		return errors.New("agent: a non-head node needs an address to serve on")
	}
	return nil
}

// ParseRange reads a block range written as "start:end", half-open.
func ParseRange(s string) (shard.Range, error) {
	start, end, ok := strings.Cut(s, ":")
	if !ok {
		return shard.Range{}, fmt.Errorf("agent: block range %q should look like start:end", s)
	}

	lo, err := strconv.Atoi(strings.TrimSpace(start))
	if err != nil {
		return shard.Range{}, fmt.Errorf("agent: block range %q: %w", s, err)
	}
	hi, err := strconv.Atoi(strings.TrimSpace(end))
	if err != nil {
		return shard.Range{}, fmt.Errorf("agent: block range %q: %w", s, err)
	}
	if hi <= lo {
		return shard.Range{}, fmt.Errorf("agent: block range %q is empty", s)
	}

	return shard.Range{Start: lo, End: hi}, nil
}

// ServeNode loads a shard and serves it until the listener closes. It is what a
// non-head process runs.
func ServeNode(spec NodeSpec) (*Server, error) {
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

	srv, err := Listen(s, spec.Listen)
	if err != nil {
		return nil, err
	}

	slog.Info("shard node serving",
		"blocks", spec.Blocks, "role", spec.Assignment().Role, "address", srv.Addr())

	return srv, nil
}

// DialStages connects to the downstream nodes, in pipeline order.
//
// Nodes are dialled with a retry window because a cluster starts as several
// processes at once and the head has no way to know when its peers finished
// loading. Loading a shard reads hundreds of megabytes, so the first connection
// attempt routinely arrives too early.
func DialStages(addresses []string, blocks []shard.Range, timeout time.Duration) ([]*Stage, error) {
	var stages []*Stage

	for i, addr := range addresses {
		deadline := time.Now().Add(timeout)

		var (
			s   *Stage
			err error
		)
		for {
			label := ""
			if i < len(blocks) {
				label = blocks[i].String()
			}
			s, err = Dial(fmt.Sprintf("stage%d", i+1), addr, label, 2*time.Second)
			if err == nil || time.Now().After(deadline) {
				break
			}
			time.Sleep(250 * time.Millisecond)
		}
		if err != nil {
			for _, done := range stages {
				_ = done.Close()
			}
			return nil, fmt.Errorf("agent: stage %d at %s did not come up within %s: %w", i+1, addr, timeout, err)
		}

		slog.Info("connected to stage", "stage", i+1, "address", addr)
		stages = append(stages, s)
	}

	return stages, nil
}
