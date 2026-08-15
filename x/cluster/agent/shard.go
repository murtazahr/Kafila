// Package agent runs one stage of a split model.
//
// A shard holds a contiguous range of transformer blocks and nothing else it
// does not need. It receives a hidden state, runs its blocks, and returns the
// result. The head is the exception: it starts from token ids because it owns
// the embedding, and it finishes the loop because it owns the output projection
// and the sampler.
//
// Everything MLX touches happens on one locked OS thread. That is not tidiness:
// MLX's CUDA streams are thread-local, and evaluating an array from a goroutine
// that has migrated fails with "no Stream(gpu, 0) in current thread". Metal
// tolerates it, which is exactly why it has to be enforced rather than assumed.
package agent

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// Shard is one stage's model and cache state.
type Shard struct {
	Model      base.Model
	Assignment shard.Assignment

	caches []cache.Cache
	thread *mlxthread.Thread

	// offset is how many tokens this shard's caches hold. The coordinator
	// compares it across stages: they must agree, because the offset drives
	// RoPE and mask construction and a divergence produces plausible-looking
	// wrong output rather than an error.
	offset int
}

// Config describes the stage to load.
type Config struct {
	ModelName  string
	Assignment shard.Assignment
	Model      shard.Model
}

// Load builds the shard: it reads only the tensors this stage owns, binds them
// to a model sized to the stage's block count, and allocates caches for those
// blocks alone.
//
// The model is constructed from the full checkpoint config and then told how
// many blocks it actually holds. Tensors arrive renumbered from zero, so an
// unmodified implementation binds them without knowing its absolute offset.
func Load(thread *mlxthread.Thread, cfg Config) (*Shard, error) {
	root, err := model.Open(cfg.ModelName)
	if err != nil {
		return nil, fmt.Errorf("agent: open %s: %w", cfg.ModelName, err)
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return nil, fmt.Errorf("agent: build model: %w", err)
	}

	sharded, ok := m.(base.Sharded)
	if !ok {
		return nil, fmt.Errorf("agent: %s cannot be split; its model does not implement base.Sharded", cfg.ModelName)
	}
	sharded.SetShard(cfg.Assignment.Range.Len(), cfg.Assignment.Role.IsHead(), cfg.Assignment.Role.IsTail())

	tensors, sel, err := mlxrunner.LoadShardTensors(root, cfg.Assignment, cfg.Model)
	if err != nil {
		return nil, err
	}

	if err := m.LoadWeights(tensors); err != nil {
		return nil, fmt.Errorf("agent: bind weights for %s: %w", cfg.Assignment.Range, err)
	}

	// Pin what the model kept and drop the rest. Whatever LoadWeights did not
	// bind is unreachable now, and Sweep is what stops it counting against the
	// device.
	collected := mlx.Collect(m)
	for _, arr := range collected {
		mlx.Pin(arr)
	}
	mlx.Sweep()
	mlx.Eval(collected...)

	s := &Shard{
		Model:      m,
		Assignment: cfg.Assignment,
		caches:     m.NewCaches(),
		thread:     thread,
	}

	slog.Info("shard loaded",
		"blocks", cfg.Assignment.Range, "role", cfg.Assignment.Role,
		"tensors", len(tensors), "caches", len(s.caches),
		"held_mib", sel.KeptBytes>>20, "skipped_mib", sel.SkippedBytes>>20,
		"share", fmt.Sprintf("%.1f%%", 100*sel.Fraction()))

	return s, nil
}

// Offset reports how many tokens this shard's caches hold.
func (s *Shard) Offset() int { return s.offset }

// Blocks reports the block range this shard owns.
func (s *Shard) Blocks() shard.Range { return s.Assignment.Range }

// Forward runs the shard's blocks over one batch and returns the hidden state
// to pass downstream.
//
// The batch carries either token ids, for the head, or a hidden state for every
// other stage. SeqOffsets travels with it rather than being tracked locally,
// because every stage must use the same positions and the coordinator is what
// makes them agree.
func (s *Shard) Forward(b *batch.Batch) (*mlx.Array, error) {
	var out *mlx.Array

	err := s.run(func() error {
		hidden, _ := s.Model.Forward(b, s.caches)
		if hidden == nil {
			return fmt.Errorf("agent: shard %s produced no hidden state", s.Assignment.Range)
		}
		mlx.Eval(hidden)
		out = hidden
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Advance by the query length, which is what the caches just absorbed.
	for _, n := range b.SeqQueryLens {
		s.offset += int(n)
		break
	}

	return out, nil
}

// Unembed projects a hidden state to vocabulary logits. Only the head owns the
// projection, so this fails anywhere else rather than returning something
// meaningless.
func (s *Shard) Unembed(hidden *mlx.Array) (*mlx.Array, error) {
	if !s.Assignment.Role.IsHead() {
		return nil, fmt.Errorf("agent: shard %s does not own the output projection", s.Assignment.Range)
	}

	var out *mlx.Array
	err := s.run(func() error {
		out = s.Model.Unembed(hidden)
		return nil
	})
	return out, err
}

// Reset drops cache state and returns the shard to offset zero. The coordinator
// issues this when stages disagree about where their caches rest, which is the
// only fully safe response: not every cache kind can rewind, so re-running from
// the start is the fallback that always works.
func (s *Shard) Reset() error {
	return s.run(func() error {
		for _, c := range s.caches {
			if c != nil {
				c.Free()
			}
		}
		s.caches = s.Model.NewCaches()
		s.offset = 0
		return nil
	})
}

// run executes fn on the MLX thread when one was supplied.
func (s *Shard) run(fn func() error) error {
	if s.thread == nil {
		return fn()
	}
	return s.thread.Do(context.Background(), fn)
}
