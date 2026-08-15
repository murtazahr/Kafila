#!/bin/bash
# Run a model split across several processes, one shard each.
#
# Each node is its own process with its own MLX context. That is not a
# convenience for local testing: two shards evaluating concurrently inside one
# process deadlock inside MLX, and one shard per process is the shape a real
# cluster has anyway. The only difference between this and several machines is
# what the stage addresses resolve to.
#
#   scripts/split_cluster.sh start [model] [nodes]
#   scripts/split_cluster.sh stop
#   scripts/split_cluster.sh status
#
# The head serves the runner's ordinary HTTP interface on $HEAD_PORT, so
# anything that speaks to a runner speaks to the pipeline unchanged.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA="${OLLAMA_BIN:-$ROOT/ollama}"
MODEL="${2:-qwen3-mlx:0.6b}"
NODES="${3:-3}"
HEAD_PORT="${HEAD_PORT:-9300}"
BASE_PORT="${BASE_PORT:-9301}"
RETURN_PORT="${RETURN_PORT:-9310}"

# One-way delays injected on each ring link when SIMULATE=1, standing in for a
# geography this deployment does not have. Measured public figures, roughly:
# Melbourne->Mumbai ~75ms, Mumbai->Sydney ~80ms, Sydney->Melbourne ~7ms.
#
# They are injected on the wire and reported apart from anything measured, so a
# demonstration is never mistaken for a result. Expect throughput to collapse:
# that is the point. It shows why you would not spread a decode loop across
# three continents.
SIMULATE="${SIMULATE:-0}"
LINK_DELAYS=("75ms" "80ms" "7ms")
LOG_DIR="${LOG_DIR:-/tmp/split-cluster}"
TRACE="${TRACE:-$LOG_DIR/trace.ndjson}"

stop_cluster() {
    pkill -f "ollama runner --shard" 2>/dev/null
    sleep 1
    echo "stopped"
}

# blocks_for prints "start end" for node $1 of $NODES over $2 blocks, giving the
# earlier nodes the remainder so sizes differ by at most one.
blocks_for() {
    local idx=$1 total=$2
    python3 - "$idx" "$total" "$NODES" <<'PY'
import sys
idx, total, n = (int(v) for v in sys.argv[1:4])
base, extra = divmod(total, n)
start = 0
for i in range(n):
    size = base + (1 if i < extra else 0)
    if i == idx:
        print(start, start + size)
        break
    start += size
PY
}

start_cluster() {
    [ -x "$OLLAMA" ] || { echo "no ollama binary at $OLLAMA; run: go build -o ollama ." >&2; exit 1; }

    stop_cluster >/dev/null
    mkdir -p "$LOG_DIR"
    rm -f "$LOG_DIR"/*.log "$TRACE"

    # The head needs the model's block count and tie flag to plan the split, and
    # both come from the manifest rather than being guessed.
    read -r TOTAL TIED < <("$OLLAMA" runner --shard --model "$MODEL" --describe 2>/dev/null) || true
    TOTAL="${TOTAL:-28}"
    TIED="${TIED:-true}"

    local tied_flag=""
    [ "$TIED" = "true" ] && tied_flag="--tied-embeddings"

    echo "model $MODEL: $TOTAL blocks across $NODES node(s), tied=$TIED"

    [ "$SIMULATE" = "1" ] && echo "simulated link delays: ${LINK_DELAYS[*]} (demo only, reported separately)"

    # A ring: head -> stage1 -> ... -> stageN -> head. Every node forwards to
    # the next rather than answering its caller, so the hidden state crosses
    # each link once instead of twice and the head is free the moment it has
    # dispatched.
    local stages=""
    for ((i = NODES - 1; i >= 1; i--)); do
        read -r start end < <(blocks_for "$i" "$TOTAL")
        local port=$((BASE_PORT + i - 1))
        local tail_flag="" next
        if [ "$i" -eq $((NODES - 1)) ]; then
            tail_flag="--tail"
            next="127.0.0.1:$RETURN_PORT"        # last node closes the ring
        else
            next="127.0.0.1:$((BASE_PORT + i))"
        fi

        local delay=""
        [ "$SIMULATE" = "1" ] && delay="--simulate-latency ${LINK_DELAYS[$i]}"

        "$OLLAMA" runner --shard --model "$MODEL" \
            --blocks "$start:$end" --total-blocks "$TOTAL" $tied_flag $tail_flag \
            --listen "127.0.0.1:$port" --next "$next" $delay \
            > "$LOG_DIR/stage$i.log" 2>&1 &

        echo "  stage $i  blocks [$start,$end)  127.0.0.1:$port -> $next"
        stages="127.0.0.1:$port=$start:$end${stages:+,$stages}"
    done

    read -r start end < <(blocks_for 0 "$TOTAL")
    local head_delay=""
    [ "$SIMULATE" = "1" ] && head_delay="--simulate-latency ${LINK_DELAYS[0]}"

    "$OLLAMA" runner --shard --model "$MODEL" \
        --blocks "$start:$end" --total-blocks "$TOTAL" $tied_flag --head \
        --stages "$stages" --next "127.0.0.1:$BASE_PORT" \
        --return-listen "127.0.0.1:$RETURN_PORT" $head_delay \
        --port "$HEAD_PORT" --trace "$TRACE" \
        > "$LOG_DIR/head.log" 2>&1 &

    echo "  head     blocks [$start,$end)  127.0.0.1:$HEAD_PORT -> 127.0.0.1:$BASE_PORT"
    echo "  ring closes at 127.0.0.1:$RETURN_PORT"
    echo
    echo "logs  $LOG_DIR"
    echo "trace $TRACE"
    echo "wait for readiness: curl -s 127.0.0.1:$HEAD_PORT/v1/status"
}

status_cluster() {
    if ! curl -s --max-time 3 "http://127.0.0.1:$HEAD_PORT/v1/status" >/dev/null; then
        echo "head not responding on $HEAD_PORT"
        pgrep -fl "ollama runner --shard" | sed 's/^/  /' || echo "  no shard processes"
        return 1
    fi
    curl -s "http://127.0.0.1:$HEAD_PORT/v1/topology"
    echo
}

case "${1:-start}" in
    start)  start_cluster ;;
    stop)   stop_cluster ;;
    status) status_cluster ;;
    *)      echo "usage: $0 {start|stop|status} [model] [nodes]" >&2; exit 2 ;;
esac
