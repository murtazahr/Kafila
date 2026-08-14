package cluster

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/shard"
)

func testRegistry(names ...string) *Registry {
	r := NewRegistry()
	for _, n := range names {
		if err := r.Add(Node{Name: n}); err != nil {
			panic(err)
		}
	}
	return r
}

func TestRegistryPreservesOrder(t *testing.T) {
	r := testRegistry("charlie", "alpha", "bravo")

	got := r.Nodes()
	want := []string{"charlie", "alpha", "bravo"}
	for i := range want {
		if got[i].Name != want[i] {
			t.Errorf("position %d = %q, want %q", i, got[i].Name, want[i])
		}
	}
}

// Re-adding a node updates it without moving it, so a node reporting fresh
// memory figures does not silently become the tail of the pipeline.
func TestRegistryAddKeepsPosition(t *testing.T) {
	r := testRegistry("a", "b", "c")

	if err := r.Add(Node{Name: "a", FreeMemory: 999}); err != nil {
		t.Fatal(err)
	}

	if r.Len() != 3 {
		t.Fatalf("re-add changed the node count to %d", r.Len())
	}
	nodes := r.Nodes()
	if nodes[0].Name != "a" {
		t.Errorf("re-added node moved to position of %q", nodes[0].Name)
	}
	if nodes[0].FreeMemory != 999 {
		t.Errorf("re-add did not update the node: free memory %d", nodes[0].FreeMemory)
	}
}

func TestRegistryRemoveReindexes(t *testing.T) {
	r := testRegistry("a", "b", "c")

	if !r.Remove("b") {
		t.Fatal("Remove reported the node absent")
	}
	if r.Remove("b") {
		t.Error("Remove reported a second removal as successful")
	}

	// The survivors must still be findable, which they are not if the name
	// index still points at pre-removal positions.
	for _, name := range []string{"a", "c"} {
		n, ok := r.Get(name)
		if !ok {
			t.Fatalf("%q vanished after removing a different node", name)
		}
		if n.Name != name {
			t.Errorf("Get(%q) returned %q; the index is stale", name, n.Name)
		}
	}
}

func TestRegistryRejectsUnnamed(t *testing.T) {
	if err := NewRegistry().Add(Node{}); err == nil {
		t.Error("registered a node with no name")
	}
}

func TestRegistryIsConcurrencySafe(t *testing.T) {
	r := NewRegistry()

	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = r.Add(Node{Name: string(rune('a' + i%26))})
			r.Nodes()
			r.Get("a")
		}()
	}
	wg.Wait()

	if r.Len() != 26 {
		t.Errorf("got %d distinct nodes, want 26", r.Len())
	}
}

func TestNodeIsLocal(t *testing.T) {
	if !(Node{Name: "a"}).IsLocal() {
		t.Error("a node with no address should be local")
	}
	if (Node{Name: "a", Address: "10.0.0.2:9000"}).IsLocal() {
		t.Error("a node with an address should not be local")
	}
}

func TestBuildAssignsInRegistrationOrder(t *testing.T) {
	r := testRegistry("head", "middle", "tail")

	p, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if p.Len() != 3 {
		t.Fatalf("got %d stages, want 3", p.Len())
	}
	if p.Head().Node.Name != "head" {
		t.Errorf("head stage is %q", p.Head().Node.Name)
	}
	if p.Tail().Node.Name != "tail" {
		t.Errorf("tail stage is %q", p.Tail().Node.Name)
	}

	if !p.Head().Role().IsHead() || p.Head().Role().IsTail() {
		t.Errorf("head role = %s", p.Head().Role())
	}
	if !p.Tail().Role().IsTail() || p.Tail().Role().IsHead() {
		t.Errorf("tail role = %s", p.Tail().Role())
	}
	if mid := p.Stages[1].Role(); mid.IsHead() || mid.IsTail() {
		t.Errorf("middle role = %s, want neither end", mid)
	}

	// 28 blocks over 3 nodes: earlier stages take the remainder.
	want := []shard.Range{{Start: 0, End: 10}, {Start: 10, End: 19}, {Start: 19, End: 28}}
	for i, w := range want {
		if got := p.Stages[i].Blocks(); got != w {
			t.Errorf("stage %d blocks = %s, want %s", i, got, w)
		}
	}
}

// The single-node plan is what Stage 1 runs on: it must be a valid plan whose
// one stage carries both roles and the whole model.
func TestBuildSingleNode(t *testing.T) {
	p, err := Build(testRegistry("solo"), shard.Model{Blocks: 28})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if !p.IsSingleNode() {
		t.Error("one node did not produce a single-node plan")
	}
	if !p.IsLocal() {
		t.Error("a node with no address produced a non-local plan")
	}

	s := p.Stages[0]
	if !s.Role().IsHead() || !s.Role().IsTail() {
		t.Errorf("solo stage role = %s, want head+tail", s.Role())
	}
	if want := (shard.Range{Start: 0, End: 28}); s.Blocks() != want {
		t.Errorf("solo stage blocks = %s, want %s", s.Blocks(), want)
	}
}

func TestBuildErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		reg   *Registry
		model shard.Model
	}{
		"no nodes":        {NewRegistry(), shard.Model{Blocks: 28}},
		"no blocks":       {testRegistry("a"), shard.Model{Blocks: 0}},
		"negative blocks": {testRegistry("a"), shard.Model{Blocks: -1}},
		"more nodes than blocks": {
			testRegistry("a", "b", "c", "d", "e"), shard.Model{Blocks: 3},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Build(tc.reg, tc.model); err == nil {
				t.Error("Build accepted an impossible request")
			}
		})
	}
}

func TestPlanIsLocalDetectsRemote(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(Node{Name: "head"})
	_ = r.Add(Node{Name: "tail", Address: "10.0.0.2:9000"})

	p, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}
	if p.IsLocal() {
		t.Error("a plan containing an addressed node reported itself local")
	}
}

func TestValidateRejectsBrokenPlans(t *testing.T) {
	for name, p := range map[string]*Plan{
		"empty": {Model: shard.Model{Blocks: 28}},
		"unnamed node": {
			Model: shard.Model{Blocks: 28},
			Stages: []Stage{
				{Node: Node{}, Assignment: shard.Assignment{Range: shard.Range{Start: 0, End: 28}, Role: shard.Head | shard.Tail}},
			},
		},
		"duplicate node": {
			Model: shard.Model{Blocks: 28},
			Stages: []Stage{
				{Node: Node{Name: "a"}, Assignment: shard.Assignment{Range: shard.Range{Start: 0, End: 14}, Role: shard.Head}},
				{Node: Node{Name: "a"}, Assignment: shard.Assignment{Range: shard.Range{Start: 14, End: 28}, Role: shard.Tail}},
			},
		},
		"gap in coverage": {
			Model: shard.Model{Blocks: 28},
			Stages: []Stage{
				{Node: Node{Name: "a"}, Assignment: shard.Assignment{Range: shard.Range{Start: 0, End: 12}, Role: shard.Head}},
				{Node: Node{Name: "b"}, Assignment: shard.Assignment{Range: shard.Range{Start: 14, End: 28}, Role: shard.Tail}},
			},
		},
		"short coverage": {
			Model: shard.Model{Blocks: 28},
			Stages: []Stage{
				{Node: Node{Name: "a"}, Assignment: shard.Assignment{Range: shard.Range{Start: 0, End: 14}, Role: shard.Head}},
				{Node: Node{Name: "b"}, Assignment: shard.Assignment{Range: shard.Range{Start: 14, End: 20}, Role: shard.Tail}},
			},
		},
		"roles at the wrong end": {
			Model: shard.Model{Blocks: 28},
			Stages: []Stage{
				{Node: Node{Name: "a"}, Assignment: shard.Assignment{Range: shard.Range{Start: 0, End: 14}, Role: shard.Tail}},
				{Node: Node{Name: "b"}, Assignment: shard.Assignment{Range: shard.Range{Start: 14, End: 28}, Role: shard.Head}},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := p.Validate(); err == nil {
				t.Error("Validate accepted a broken plan")
			}
		})
	}
}

func TestPlanNavigation(t *testing.T) {
	p, err := Build(testRegistry("a", "b", "c"), shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}

	next, ok := p.Next(0)
	if !ok || next.Node.Name != "b" {
		t.Errorf("Next(0) = %v, %v; want b", next.Node.Name, ok)
	}
	if _, ok := p.Next(2); ok {
		t.Error("Next reported a stage downstream of the tail")
	}
	if _, ok := p.Next(-1); ok {
		t.Error("Next accepted a negative index")
	}

	if s, ok := p.StageFor("b"); !ok || s.Blocks().Start != 10 {
		t.Errorf("StageFor(b) = %v, %v", s.Blocks(), ok)
	}
	if _, ok := p.StageFor("nope"); ok {
		t.Error("StageFor found a node that is not in the plan")
	}
}

func TestPlanLibraries(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(Node{Name: "a", Library: "Metal"})
	_ = r.Add(Node{Name: "b", Library: "CUDA"})
	_ = r.Add(Node{Name: "c", Library: "Metal"})

	p, err := Build(r, shard.Model{Blocks: 28})
	if err != nil {
		t.Fatal(err)
	}

	got := p.Libraries()
	if len(got) != 2 || got[0] != "CUDA" || got[1] != "Metal" {
		t.Errorf("Libraries = %v, want [CUDA Metal]", got)
	}
}

// A plan is handed to nodes over the wire, so it has to survive the trip with
// its ranges and roles intact.
func TestPlanRoundTripsThroughJSON(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(Node{Name: "head", Library: "Metal", TotalMemory: 1 << 34})
	_ = r.Add(Node{Name: "tail", Address: "10.0.0.2:9000", Library: "CUDA"})

	want, err := Build(r, shard.Model{Blocks: 28, TiedEmbeddings: true})
	if err != nil {
		t.Fatal(err)
	}

	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}

	var got Plan
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}

	if err := got.Validate(); err != nil {
		t.Fatalf("round-tripped plan is invalid: %v", err)
	}
	if got.Model != want.Model {
		t.Errorf("model = %+v, want %+v", got.Model, want.Model)
	}
	for i := range want.Stages {
		if got.Stages[i] != want.Stages[i] {
			t.Errorf("stage %d = %+v, want %+v", i, got.Stages[i], want.Stages[i])
		}
	}
}

func TestRegistryByFreeMemory(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(Node{Name: "small", FreeMemory: 1 << 30})
	_ = r.Add(Node{Name: "large", FreeMemory: 64 << 30})
	_ = r.Add(Node{Name: "medium", FreeMemory: 16 << 30})

	got := r.ByFreeMemory()
	want := []string{"large", "medium", "small"}
	for i := range want {
		if got[i].Name != want[i] {
			t.Errorf("position %d = %q, want %q", i, got[i].Name, want[i])
		}
	}

	// Ordering by capability must not disturb the pipeline order the planner
	// actually uses.
	if r.Nodes()[0].Name != "small" {
		t.Error("ByFreeMemory reordered the registry itself")
	}
}
