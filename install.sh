#!/usr/bin/env bash
# Qwen3.5-9B Defiant - local install
#
# Sets up Ollama >= 0.32 (required: the GGUFs declare architecture "qwen35"),
# downloads quants matched to this machine's VRAM, verifies them, registers them
# with Ollama and installs systemd user services.
#
# No sudo required - everything lands under $HOME.
set -euo pipefail

REPO=DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF
BASE=https://huggingface.co/$REPO/resolve/main
PREFIX=$HOME/.local/ollama-new
MODELS=${MODELS_DIR:-$(cd "$(dirname "$0")" && pwd)/models}
PORT_OLLAMA=${PORT_OLLAMA:-11435}
PORT_CHAT=${PORT_CHAT:-8181}
STEM=Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX

say(){ printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

for c in curl tar awk sed; do command -v $c >/dev/null || die "missing required command: $c"; done
command -v zstd >/dev/null || die "missing 'zstd' - install it (apt install zstd) and re-run"

# ---------------------------------------------------------------- hardware
say "Detecting hardware"
VRAM_MB=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "  GPU:  $GPU  (${VRAM_MB} MiB VRAM)"
else
  warn "no working NVIDIA GPU detected - will run on CPU (slow)"
  GPU="none"
fi
RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
echo "  RAM:  ${RAM_GB} GiB"
echo "  Disk: $(df -h "$(dirname "$MODELS")" | awk 'NR==2{print $4}') free"

# Pick quants by VRAM. Speed is dominated by whether every layer fits in VRAM;
# once layers spill to CPU, throughput drops roughly linearly.
declare -a FILES DESC
if   [ "$VRAM_MB" -ge 22000 ]; then
  FILES=("$STEM-MTP-Q8_0.gguf");  DESC="Q8_0 (best quality; fits comfortably)"
elif [ "$VRAM_MB" -ge 14000 ]; then
  FILES=("$STEM-Q6_K.gguf");      DESC="Q6_K (high quality, full offload)"
elif [ "$VRAM_MB" -ge 9000 ]; then
  FILES=("$STEM-Q4_K_M.gguf");    DESC="Q4_K_M (good quality, long context)"
elif [ "$VRAM_MB" -ge 7000 ]; then
  FILES=("$STEM-Q4_K_M.gguf" "$STEM-IQ3_M.gguf")
  DESC="Q4_K_M (quality, <=16K ctx) + IQ3_M (64K ctx)"
else
  FILES=("$STEM-IQ3_M.gguf");     DESC="IQ3_M (smallest practical)"
fi
[ "${WITH_VISION:-1}" = 1 ] && FILES+=("mmproj-F16.gguf")
[ "${WITH_ARCHIVE:-0}" = 1 ] && FILES+=("$STEM-MTP-Q8_0.gguf")

say "Selected for this machine: $DESC"
printf '  %s\n' "${FILES[@]}"
[ "${YES:-0}" = 1 ] || { read -rp $'\nProceed? [y/N] ' a; [[ $a =~ ^[Yy] ]] || exit 1; }

# ---------------------------------------------------------------- ollama
NEED_OLLAMA=1
if [ -x "$PREFIX/dist/bin/ollama" ]; then
  V=$("$PREFIX/dist/bin/ollama" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  MAJ=${V%%.*}; MIN=$(echo "$V" | cut -d. -f2)
  if [ "${MAJ:-0}" -gt 0 ] || [ "${MIN:-0}" -ge 32 ]; then
    NEED_OLLAMA=0; say "Ollama $V already present"
  fi
fi
if [ "$NEED_OLLAMA" = 1 ]; then
  say "Installing Ollama (user-local; system install untouched)"
  mkdir -p "$PREFIX"; cd "$PREFIX"
  curl -fL --progress-bar -o ollama.tar.zst \
    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
  curl -fsSL -o sha256sum.txt \
    https://github.com/ollama/ollama/releases/latest/download/sha256sum.txt
  EXP=$(grep 'linux-amd64.tar.zst' sha256sum.txt | awk '{print $1}')
  GOT=$(sha256sum ollama.tar.zst | awk '{print $1}')
  [ "$EXP" = "$GOT" ] || die "ollama checksum mismatch (expected $EXP got $GOT)"
  mkdir -p dist && zstd -dc ollama.tar.zst | tar -xf - -C dist && rm -f ollama.tar.zst
  echo "  installed $("$PREFIX/dist/bin/ollama" --version 2>&1 | grep -oE '[0-9.]+' | head -1)"
fi
OLLAMA=$PREFIX/dist/bin/ollama

# ---------------------------------------------------------------- weights
say "Downloading model files to $MODELS"
mkdir -p "$MODELS"
for f in "${FILES[@]}"; do
  if [ -s "$MODELS/$f" ]; then echo "  have  $f"; continue; fi
  echo "  fetch $f"
  curl -fL --progress-bar -o "$MODELS/$f.part" "$BASE/$f"
  mv "$MODELS/$f.part" "$MODELS/$f"
done

if [ -f "$MODELS/SHA256SUMS.txt" ]; then
  say "Verifying checksums"
  ( cd "$MODELS" && grep -F -f <(printf '%s\n' "${FILES[@]}") SHA256SUMS.txt \
      | sha256sum -c - ) || die "checksum verification FAILED - delete the bad file and re-run"
else
  warn "SHA256SUMS.txt not found; skipping verification"
fi

# ---------------------------------------------------------------- register
say "Starting Ollama on :$PORT_OLLAMA"
export OLLAMA_HOST=127.0.0.1:$PORT_OLLAMA
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
if ! curl -sf --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null; then
  "$OLLAMA" serve >/tmp/ollama-install.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 2
  done
fi
curl -sf --max-time 3 "http://$OLLAMA_HOST/api/version" >/dev/null \
  || die "ollama failed to start; see /tmp/ollama-install.log"

register(){ # name file [mmproj]
  local name=$1 gguf=$2 proj=${3:-}
  [ -s "$MODELS/$gguf" ] || return 0
  local mf; mf=$(mktemp)
  echo "FROM $MODELS/$gguf" > "$mf"
  [ -n "$proj" ] && [ -s "$MODELS/$proj" ] && echo "FROM $MODELS/$proj" >> "$mf"
  echo "  register $name"
  "$OLLAMA" create "$name" -f "$mf" >/dev/null 2>&1 || warn "failed to register $name"
  rm -f "$mf"
}
say "Registering models with Ollama"
register qwen35-fast    "$STEM-IQ3_M.gguf"
register qwen35-quality "$STEM-Q4_K_M.gguf"
register qwen35-q6      "$STEM-Q6_K.gguf"
register qwen35-q8      "$STEM-MTP-Q8_0.gguf"
# vision layers onto whichever base is present, smallest first for VRAM headroom
for b in "$STEM-IQ3_M.gguf" "$STEM-Q4_K_M.gguf" "$STEM-Q6_K.gguf"; do
  [ -s "$MODELS/$b" ] && { register qwen35-vision "$b" mmproj-F16.gguf; break; }
done

# Older hand-imported tags: alias them to the canonical names so chat.py and
# run-qwen35.sh resolve the same models on every machine. `ollama cp` copies
# metadata, not blobs, so this costs nothing on disk.
have_tag(){ "$OLLAMA" list 2>/dev/null | awk 'NR>1{sub(/:.*/,"",$1); print $1}' | grep -qx "$1"; }
alias_legacy(){ # canonical legacy
  have_tag "$1" && return 0
  have_tag "$2" || return 0
  echo "  alias $2 -> $1"
  "$OLLAMA" cp "$2" "$1" >/dev/null 2>&1 || warn "could not alias $2 -> $1"
}
say "Reconciling model names"
alias_legacy qwen35-fast    qwen3_5_9b_iq3
alias_legacy qwen35-quality qwen35-defiant-q4km

# The API serves aliases, so an alias with no backing tag is a 404 on every
# request. Catch that here rather than at first use.
say "Verifying the aliases the API serves"
MISSING=0
for a in qwen35-fast qwen35-quality qwen35-vision; do
  if have_tag "$a"; then echo "  ok       $a"
  else warn "missing  $a - the API will 404 for it"; MISSING=1; fi
done
[ "$MISSING" = 0 ] || warn "run '$OLLAMA list' and check the register step above"

# ---------------------------------------------------------------- services
say "Installing systemd user services"
mkdir -p "$HOME/.config/systemd/user"
sed -e "s|__PREFIX__|$PREFIX|g" -e "s|__PORT__|$PORT_OLLAMA|g" \
    "$(dirname "$0")/systemd/ollama-qwen35.service" \
    > "$HOME/.config/systemd/user/ollama-qwen35.service"
sed -e "s|__DIR__|$(cd "$(dirname "$0")" && pwd)|g" -e "s|__PORT__|$PORT_CHAT|g" \
    "$(dirname "$0")/systemd/qwen35-chat.service" \
    > "$HOME/.config/systemd/user/qwen35-chat.service"
systemctl --user daemon-reload
systemctl --user enable --now ollama-qwen35.service qwen35-chat.service
loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable linger; services start at login rather than boot"

sleep 5
say "Done"
curl -sf --max-time 5 "http://127.0.0.1:$PORT_CHAT/healthz" >/dev/null \
  && echo "  chat UI     http://127.0.0.1:$PORT_CHAT/" \
  || warn "chat service not responding yet - check: journalctl --user -u qwen35-chat -n 50"
echo "  OpenAI API  http://127.0.0.1:$PORT_CHAT/v1"
echo "  models:     $("$OLLAMA" list 2>/dev/null | awk 'NR>1{printf "%s ", $1}')"
