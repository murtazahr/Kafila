// Package cluster describes the machines a model is split across and how the
// split is assigned to them.
//
// It deliberately knows nothing about how a shard executes or how activations
// move between nodes. It answers only two questions: which nodes exist, and
// which blocks each one owns. Keeping that separate from execution is what lets
// placement policy become a research variable later without disturbing the
// transport or the trace plane.
package cluster

import (
	"fmt"
	"sort"
	"sync"
)

// Node is a machine that can host a shard.
type Node struct {
	// Name identifies the node in plans and traces. Stable across restarts.
	Name string `json:"name"`

	// Address is the host:port of the node's shard agent. Empty for a node
	// that runs in this process.
	Address string `json:"address,omitempty"`

	// Library is the backend the node's device uses, as reported by
	// discovery: "Metal", "CUDA". Nodes are not required to match, but a
	// plan that mixes them is worth noticing.
	Library string `json:"library,omitempty"`

	// TotalMemory and FreeMemory are the device's memory in bytes, as last
	// observed. Advisory: they inform placement but are not a guarantee, and
	// a node can refuse a shard it cannot fit.
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
}

// IsLocal reports whether the node runs in this process. A local node needs no
// transport, which is what makes a single-node cluster exercise the whole
// control path without a network.
func (n Node) IsLocal() bool { return n.Address == "" }

func (n Node) String() string {
	if n.IsLocal() {
		return n.Name + " (local)"
	}
	return n.Name + " @ " + n.Address
}

// Registry is the set of nodes available to host shards.
//
// Order matters: it is the pipeline order a plan will use, so the first
// registered node becomes the head. Choosing that order by capability rather
// than by registration is R4 and deliberately not done here.
type Registry struct {
	mu    sync.RWMutex
	nodes []Node
	index map[string]int
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{index: map[string]int{}}
}

// Add registers a node. Re-adding a name updates that node in place, keeping
// its position in the pipeline order, so a node reporting fresh memory figures
// does not silently move to the end of the pipeline.
func (r *Registry) Add(n Node) error {
	if n.Name == "" {
		return fmt.Errorf("cluster: node needs a name")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if i, ok := r.index[n.Name]; ok {
		r.nodes[i] = n
		return nil
	}

	r.index[n.Name] = len(r.nodes)
	r.nodes = append(r.nodes, n)
	return nil
}

// Remove drops a node, reporting whether it was present.
func (r *Registry) Remove(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	i, ok := r.index[name]
	if !ok {
		return false
	}

	r.nodes = append(r.nodes[:i], r.nodes[i+1:]...)
	delete(r.index, name)
	for j := i; j < len(r.nodes); j++ {
		r.index[r.nodes[j].Name] = j
	}
	return true
}

// Get returns a node by name.
func (r *Registry) Get(name string) (Node, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	i, ok := r.index[name]
	if !ok {
		return Node{}, false
	}
	return r.nodes[i], true
}

// Nodes returns the registered nodes in pipeline order.
func (r *Registry) Nodes() []Node {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]Node(nil), r.nodes...)
}

// Len returns the number of registered nodes.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.nodes)
}

// ByFreeMemory returns the nodes ordered by free memory, most first.
//
// Not used by the default planner, which follows registration order. It exists
// so a capability-aware planner has the ordering available without reaching
// into the registry's internals.
func (r *Registry) ByFreeMemory() []Node {
	ns := r.Nodes()
	sort.SliceStable(ns, func(i, j int) bool { return ns[i].FreeMemory > ns[j].FreeMemory })
	return ns
}
