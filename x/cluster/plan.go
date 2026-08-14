package cluster

import (
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

// Stage is one node's place in the pipeline.
type Stage struct {
	Node       Node             `json:"node"`
	Assignment shard.Assignment `json:"assignment"`
}

// Blocks renders the stage's block range.
func (s Stage) Blocks() shard.Range { return s.Assignment.Range }

// Role renders the stage's role.
func (s Stage) Role() shard.Role { return s.Assignment.Role }

// Plan is a complete assignment of a model's blocks to nodes, in pipeline
// order: stage 0 is the head and the last stage is the tail.
//
// A plan is the unit that gets distributed to nodes, so it is serializable and
// self-describing — a node can be handed its own stage and the plan it belongs
// to, and know everything it needs about its neighbours.
type Plan struct {
	Model  shard.Model `json:"model"`
	Stages []Stage     `json:"stages"`
}

// Build assigns a model's blocks across the registry's nodes in registration
// order, splitting evenly.
//
// Even splitting is the deliberate baseline: it ignores per-layer cost, node
// capability, and the weight the ring topology pins to the head, all of which
// matter and none of which are free to get right. See R4. Its value is being
// obvious enough that a smarter planner has something unambiguous to beat.
func Build(r *Registry, m shard.Model) (*Plan, error) {
	nodes := r.Nodes()
	if len(nodes) == 0 {
		return nil, fmt.Errorf("cluster: no nodes registered")
	}
	if m.Blocks <= 0 {
		return nil, fmt.Errorf("cluster: model has %d blocks", m.Blocks)
	}
	if len(nodes) > m.Blocks {
		return nil, fmt.Errorf("cluster: %d nodes for %d blocks; a shard would own none", len(nodes), m.Blocks)
	}

	as, err := m.Split(len(nodes))
	if err != nil {
		return nil, err
	}

	p := &Plan{Model: m, Stages: make([]Stage, len(nodes))}
	for i := range nodes {
		p.Stages[i] = Stage{Node: nodes[i], Assignment: as[i]}
	}

	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

// Validate checks that the plan is executable.
//
// A plan that tiles the model incorrectly does not fail loudly at run time — it
// produces wrong output — so this is worth being strict about. It re-runs the
// shard-level checks and adds the ones that only make sense once nodes are
// attached: distinct nodes, and a coherent pipeline order.
func (p *Plan) Validate() error {
	if p == nil || len(p.Stages) == 0 {
		return fmt.Errorf("cluster: empty plan")
	}

	as := make([]shard.Assignment, len(p.Stages))
	seen := make(map[string]bool, len(p.Stages))
	for i, s := range p.Stages {
		if s.Node.Name == "" {
			return fmt.Errorf("cluster: stage %d has an unnamed node", i)
		}
		if seen[s.Node.Name] {
			return fmt.Errorf("cluster: node %q appears in more than one stage", s.Node.Name)
		}
		seen[s.Node.Name] = true
		as[i] = s.Assignment
	}

	return shard.Validate(as, p.Model.Blocks)
}

// Head is the stage that owns the embedding, the output projection and the
// sampler.
func (p *Plan) Head() Stage { return p.Stages[0] }

// Tail is the stage that owns the final norm.
func (p *Plan) Tail() Stage { return p.Stages[len(p.Stages)-1] }

// Len returns the number of stages.
func (p *Plan) Len() int { return len(p.Stages) }

// IsSingleNode reports whether the whole model runs in one place. This is the
// degenerate plan Stage 1 uses: it exercises the entire control path — planning,
// selection, tracing, the façade — while inference itself stays byte-identical
// to the unsharded runner, so a failure implicates the new machinery rather
// than the split.
func (p *Plan) IsSingleNode() bool { return len(p.Stages) == 1 }

// IsLocal reports whether every stage runs in this process.
func (p *Plan) IsLocal() bool {
	for _, s := range p.Stages {
		if !s.Node.IsLocal() {
			return false
		}
	}
	return true
}

// StageFor returns the stage assigned to a node.
func (p *Plan) StageFor(node string) (Stage, bool) {
	for _, s := range p.Stages {
		if s.Node.Name == node {
			return s, true
		}
	}
	return Stage{}, false
}

// Next returns the stage downstream of the given one, and false at the tail.
func (p *Plan) Next(i int) (Stage, bool) {
	if i < 0 || i+1 >= len(p.Stages) {
		return Stage{}, false
	}
	return p.Stages[i+1], true
}

// Libraries returns the distinct backends the plan spans, sorted. More than one
// means a heterogeneous pipeline, which is supported but worth surfacing: the
// stages will not have comparable performance, and an even block split will be
// a poor fit.
func (p *Plan) Libraries() []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range p.Stages {
		lib := s.Node.Library
		if lib == "" || seen[lib] {
			continue
		}
		seen[lib] = true
		out = append(out, lib)
	}
	sort.Strings(out)
	return out
}

func (p *Plan) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d blocks across %d node(s)", p.Model.Blocks, len(p.Stages))
	if p.Model.TiedEmbeddings {
		b.WriteString(", tied embeddings")
	}
	for _, s := range p.Stages {
		fmt.Fprintf(&b, "\n  %-10s %-9s %s", s.Node.Name, s.Assignment.Range, s.Assignment.Role)
		if !s.Node.IsLocal() {
			b.WriteString(" @ " + s.Node.Address)
		}
	}
	return b.String()
}
