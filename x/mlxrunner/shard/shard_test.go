package shard

import "testing"

func TestSplit(t *testing.T) {
	for name, tc := range map[string]struct {
		blocks int
		n      int
		want   []Range
	}{
		"single shard":    {32, 1, []Range{{0, 32}}},
		"even halves":     {32, 2, []Range{{0, 16}, {16, 32}}},
		"uneven thirds":   {32, 3, []Range{{0, 11}, {11, 22}, {22, 32}}},
		"uneven quarters": {30, 4, []Range{{0, 8}, {8, 16}, {16, 23}, {23, 30}}},
		"one each":        {3, 3, []Range{{0, 1}, {1, 2}, {2, 3}}},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := Split(tc.blocks, tc.n)
			if err != nil {
				t.Fatalf("Split(%d, %d) failed: %v", tc.blocks, tc.n, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("Split returned %d assignments, want %d", len(got), len(tc.want))
			}
			for i, a := range got {
				if a.Range != tc.want[i] {
					t.Errorf("assignment %d = %s, want %s", i, a.Range, tc.want[i])
				}
			}
			if err := Validate(got, tc.blocks); err != nil {
				t.Errorf("Split produced an invalid plan: %v", err)
			}
		})
	}
}

func TestSplitRoles(t *testing.T) {
	got, err := Split(32, 3)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}

	if !got[0].Role.IsHead() || got[0].Role.IsTail() {
		t.Errorf("first shard role = %s, want head only", got[0].Role)
	}
	if got[1].Role.IsHead() || got[1].Role.IsTail() {
		t.Errorf("middle shard role = %s, want middle", got[1].Role)
	}
	if got[2].Role.IsHead() || !got[2].Role.IsTail() {
		t.Errorf("last shard role = %s, want tail only", got[2].Role)
	}
}

// A single shard is both ends of the pipeline; this is the cluster-of-one case
// that must behave exactly like unsharded inference.
func TestSplitSingleShardIsHeadAndTail(t *testing.T) {
	got, err := Split(32, 1)
	if err != nil {
		t.Fatalf("Split failed: %v", err)
	}
	if !got[0].Role.IsHead() || !got[0].Role.IsTail() {
		t.Errorf("single shard role = %s, want head+tail", got[0].Role)
	}
}

func TestSplitErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		blocks int
		n      int
	}{
		"zero blocks":     {0, 2},
		"negative blocks": {-1, 2},
		"zero shards":     {32, 0},
		"negative shards": {32, -1},
		"more shards":     {4, 8},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Split(tc.blocks, tc.n); err == nil {
				t.Errorf("Split(%d, %d) succeeded, want error", tc.blocks, tc.n)
			}
		})
	}
}

// Split must never leave a block unassigned or assign one twice, for any
// combination in the range we care about. A gap would silently drop layers
// from the model.
func TestSplitTilesExactly(t *testing.T) {
	for blocks := 1; blocks <= 80; blocks++ {
		for n := 1; n <= blocks && n <= 8; n++ {
			as, err := Split(blocks, n)
			if err != nil {
				t.Fatalf("Split(%d, %d) failed: %v", blocks, n, err)
			}

			seen := make([]int, blocks)
			for _, a := range as {
				for i := a.Range.Start; i < a.Range.End; i++ {
					seen[i]++
				}
			}
			for i, c := range seen {
				if c != 1 {
					t.Fatalf("Split(%d, %d): block %d covered %d times", blocks, n, i, c)
				}
			}
		}
	}
}

// Shard sizes must differ by at most one, or a node ends up disproportionately
// loaded and becomes the pipeline's bottleneck.
func TestSplitIsBalanced(t *testing.T) {
	for blocks := 1; blocks <= 80; blocks++ {
		for n := 1; n <= blocks && n <= 8; n++ {
			as, err := Split(blocks, n)
			if err != nil {
				t.Fatalf("Split(%d, %d) failed: %v", blocks, n, err)
			}

			lo, hi := as[0].Range.Len(), as[0].Range.Len()
			for _, a := range as {
				lo = min(lo, a.Range.Len())
				hi = max(hi, a.Range.Len())
			}
			if hi-lo > 1 {
				t.Errorf("Split(%d, %d) sizes span %d..%d", blocks, n, lo, hi)
			}
		}
	}
}

func TestValidateRejectsBadPlans(t *testing.T) {
	for name, as := range map[string][]Assignment{
		"empty": {},
		"gap": {
			{Range{0, 10}, Head},
			{Range{12, 32}, Tail},
		},
		"overlap": {
			{Range{0, 16}, Head},
			{Range{14, 32}, Tail},
		},
		"short coverage": {
			{Range{0, 16}, Head},
			{Range{16, 30}, Tail},
		},
		"empty range": {
			{Range{0, 16}, Head},
			{Range{16, 16}, 0},
			{Range{16, 32}, Tail},
		},
		"missing head": {
			{Range{0, 16}, 0},
			{Range{16, 32}, Tail},
		},
		"missing tail": {
			{Range{0, 16}, Head},
			{Range{16, 32}, 0},
		},
		"head in the middle": {
			{Range{0, 16}, Head},
			{Range{16, 32}, Head | Tail},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := Validate(as, 32); err == nil {
				t.Errorf("Validate accepted an invalid plan")
			}
		})
	}
}
