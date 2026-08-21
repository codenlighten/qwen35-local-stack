# Qwen3.5-9B Defiant — local setup

Runs `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF`
on an RTX 5060 Laptop (8 GB VRAM).

## Install on a new machine

    git clone <this-repo> qwen-new
    cd qwen-new
    ./install.sh

`install.sh` needs no sudo — everything lands under `$HOME`. It:

1. Detects GPU and VRAM, and picks quants that fit (see table below)
2. Installs a user-local Ollama >= 0.32 into `~/.local/ollama-new` (checksum-verified),
   leaving any system Ollama alone
3. Downloads the chosen GGUFs from Hugging Face and verifies them against
   `models/SHA256SUMS.txt`
4. Registers them with Ollama and installs the systemd user services
5. Enables lingering so everything starts at boot

Model weights are **not** in this repo (~25 GB); the installer fetches them.

### VRAM tiers the installer uses

| VRAM | Picks | Why |
|---|---|---|
| >= 22 GB | `MTP-Q8_0` | best quality, fits with room to spare |
| 14-22 GB | `Q6_K` | high quality, full offload |
| 9-14 GB | `Q4_K_M` | good quality with long context |
| 7-9 GB | `Q4_K_M` + `IQ3_M` | quality at <=16K, or 64K via IQ3_M |
| < 7 GB | `IQ3_M` | smallest practical |

Useful environment overrides:

    WITH_ARCHIVE=1 ./install.sh   # also fetch the 10.7 GB Q8_0 archive
    WITH_VISION=0  ./install.sh   # skip the vision projector
    YES=1          ./install.sh   # no confirmation prompt
    PORT_CHAT=9000 ./install.sh   # different port

## Requirements

Ollama **>= 0.32** — the GGUF declares architecture `qwen35`, which older builds
(including the 0.12.6 in `/usr/local/bin`) cannot load. A user-local 0.32.14 lives in
`~/.local/ollama-new/dist` and serves on port **11435**; the system install on 11434
is left untouched.

## Quick start

    ./run-qwen35.sh web      # chat UI + OpenAI API on http://127.0.0.1:8181
    ./run-qwen35.sh chat     # terminal chat, IQ3_M
    ./run-qwen35.sh hq       # terminal chat, Q4_K_M (better quality)
    ./run-qwen35.sh see      # terminal chat, vision
    ./run-qwen35.sh stop

## Autostart (systemd user services)

Both servers run as systemd **user** units and start at boot (lingering is enabled,
so they come up without anyone logging in):

    systemctl --user status  ollama-qwen35 qwen35-chat
    systemctl --user restart qwen35-chat
    systemctl --user stop    ollama-qwen35 qwen35-chat
    journalctl --user -u qwen35-chat -f

Unit files live in `~/.config/systemd/user/`. `run-qwen35.sh` still works for
ad-hoc use, but is no longer needed to bring the stack up.

## Models

| Alias | File | Max ctx at full GPU offload | Speed |
|---|---|---|---|
| `qwen35-fast`    | IQ3_M 5.3 GB   | 64K | ~34 tok/s |
| `qwen35-quality` | Q4_K_M 6.4 GB  | 16K | ~33 tok/s |
| `qwen35-vision`  | IQ3_M + mmproj-F16 | 8K | ~34 tok/s |

`MTP-Q8_0` (11 GB) is archived for future hardware — it does not fit this 8 GB card.

Speed is governed almost entirely by whether all 33 layers fit in VRAM. Once layers
spill to CPU, throughput drops roughly linearly. `OLLAMA_FLASH_ATTENTION=1` and
`OLLAMA_KV_CACHE_TYPE=q8_0` (both set by `run-qwen35.sh`) halve the KV cache and are
what make 64K fit.

## OpenAI-compatible API

Point any OpenAI client at `http://127.0.0.1:8181/v1`; no API key required.

    from openai import OpenAI
    c = OpenAI(base_url="http://127.0.0.1:8181/v1", api_key="not-needed")
    r = c.chat.completions.create(
        model="qwen35-fast",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=200)
    print(r.choices[0].message.content)

Supported: streaming, `response_format` (`json_object` and `json_schema`),
`temperature`, `top_p`, `seed`, `stop`, `max_tokens`, vision via `image_url`
(data URIs), and `/v1/models`.

### Reasoning

This is a thinking model. Reasoning is returned separately as
`message.reasoning_content` rather than inlined in `content`, so normal clients see
clean answers.

`max_tokens` bounds the **answer**, not the reasoning — the server adds a separate
reasoning allowance on top (default 2048, override with `reasoning_budget`).
Without that, a small `max_tokens` gets eaten entirely by the thinking block and
returns empty content with `finish_reason: length`.

Disable thinking per request with `extra_body={"think": False}` — much faster for
simple extraction and classification work.

## Integrity

All five GGUFs were verified byte-identical to the upstream Hugging Face hashes.
Re-check any time with:

    cd models && sha256sum -c SHA256SUMS.txt

The archived `MTP-Q8_0` was also test-loaded successfully: it reports **34** layers
rather than 33, confirming the MTP head is real and that Ollama 0.32 handles the MTP
variant. On this 8 GB card it only reaches ~0.6 tok/s (21/34 layers offloaded), which
is expected — it is stored for larger hardware, not for this laptop.

Note: `ollama create` **copies** each GGUF into `~/.ollama/models/blobs`, so imported
models exist twice on disk (once in `models/`, once in the Ollama store).
