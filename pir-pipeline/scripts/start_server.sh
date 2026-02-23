#!/bin/bash
set -e

MODEL="nvidia/Cosmos-Reason2-2B"
PORT=8000
GPU_MEM=0.75
MAX_LEN=16384
LOG="/tmp/vllm_cosmos.log"
PID_FILE="/tmp/vllm_cosmos.pid"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port)  PORT="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " Cosmos Reason2 — vLLM Server"
echo " Model: $MODEL"
echo " Port:  $PORT"
echo "============================================"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 3
    fi
fi

# Launch - CORREGIDO: backslash al final de --media-io-kwargs
echo "Starting vLLM server..."
nohup uv run vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --trust-remote-code \
    --allowed-local-media-path "$(pwd)" \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM PID: $(cat $PID_FILE)"
echo "Logs: $LOG"
echo ""
echo "Waiting for server to be ready..."

for i in $(seq 1 80); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "✓ Server ready at http://localhost:$PORT"
        echo ""
        echo "Run the pipeline:"
        echo "  python scripts/run_pipeline.py --video /path/video.mp4 --annot /path/seq.csv"
        exit 0
    fi
    printf "."
    sleep 3
done

echo ""
echo "ERROR: Server did not start within 4 minutes."
echo "Check logs: tail -f $LOG"
exit 1
