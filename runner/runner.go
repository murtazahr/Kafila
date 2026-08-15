package runner

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner"
)

func Execute(args []string) error {
	if args[0] == "runner" {
		args = args[1:]
	}

	if len(args) > 0 {
		switch args[0] {
		case "--mlx-engine":
			return mlxrunner.Execute(args[1:])
		case "--shard":
			// One node of a split model. Every node runs this and differs only
			// in flags; see x/mlxrunner/shardnode.go.
			return mlxrunner.ExecuteShard(args[1:])
		}
	}
	return fmt.Errorf("unknown runner engine, expected --mlx-engine or --shard")
}
