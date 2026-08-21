#!/usr/bin/env bash
# Qwen3.5-9B Defiant — local runner
# Needs Ollama >= 0.32 (arch "qwen35"); the distro's 0.12.6 cannot load these.
set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama-new/dist/bin/ollama}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
export OLLAMA_FLASH_ATTENTION=1     # required for the ctx figures below
export OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV cache; no quality loss worth worrying about

start() {
  if curl -sf --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null; then
    echo "already up on $OLLAMA_HOST"; return
  fi
  nohup "$OLLAMA_BIN" serve >/tmp/ollama-qwen35.log 2>&1 &
  for _ in $(seq 1 20); do
    curl -sf --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null && { echo "up on $OLLAMA_HOST"; return; }
    sleep 2
  done
  echo "server failed to start; see /tmp/ollama-qwen35.log" >&2; exit 1
}

case "${1:-chat}" in
  start) start ;;
  stop)  pkill -f "$OLLAMA_BIN serve" && echo stopped ;;
  # 64K at full GPU offload, ~34 tok/s
  chat)  start; "$OLLAMA_BIN" run qwen3_5_9b_iq3:latest ;;
  # better quality; keep ctx <= 16K to stay fully on the GPU
  hq)    start; "$OLLAMA_BIN" run qwen35-defiant-q4km ;;
  # IQ3_M + mmproj-F16; pass an image path in the prompt to use it
  see)   start; "$OLLAMA_BIN" run qwen35-vision ;;
  # web chat UI + OpenAI-compatible API on :8181
  web)   start
         pkill -f "chat.py --port 8181" 2>/dev/null || true
         echo "chat UI    http://127.0.0.1:8181/"
         echo "OpenAI API http://127.0.0.1:8181/v1"
         exec python3 "$(dirname "$0")/chat.py" --port 8181 ;;
  *) echo "usage: $0 {start|stop|chat|hq|see|web}" >&2; exit 2 ;;
esac
