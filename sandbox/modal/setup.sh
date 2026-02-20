#!/bin/bash
set -euo pipefail

# Rust cargo is installed in the image but not on PATH by default.
export PATH="$HOME/.cargo/bin:$PATH"
AGENT_KIND="${AGENT_KIND:-opencode}"

log() {
    echo "[setup] $*"
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local max_seconds="$3"
    local i

    log "waiting for ${name} (${url})"
    for ((i = 1; i <= max_seconds; i++)); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "${name} ready"
            return 0
        fi
        if (( i % 5 == 0 )); then
            log "still waiting for ${name} (${i}s)"
        fi
        sleep 1
    done

    log "ERROR: timeout waiting for ${name}"
    return 1
}

if [ -f /tmp/upload/task_setup.sh ]; then
    log "running task-specific setup"
    bash /tmp/upload/task_setup.sh
    log "done: running task-specific setup"
fi

log "starting envoi runtime on :8000"
cd /environment
python3 -m envoi.runtime --file main.py --port 8000 > /tmp/envoi.log 2>&1 &
ENVOI_PID=$!
echo "$ENVOI_PID" > /tmp/envoi.pid
log "envoi process started (pid=${ENVOI_PID})"
wait_for_http "envoi" "http://localhost:8000/schema" 120

log "initializing workspace git repo"
mkdir -p /workspace
cd /workspace
git init >/dev/null
git config user.email "agent@example.com"
git config user.name "Agent"
git commit --allow-empty -m "Initial empty commit" >/dev/null
log "done: initializing workspace git repo"

log "updating apt package metadata"
apt-get update -qq
log "done: updating apt package metadata"
log "installing ripgrep"
apt-get install -y -qq ripgrep
log "done: installing ripgrep"

if [ "$AGENT_KIND" = "opencode" ]; then
    log "installing NodeSource repository setup"
    bash -lc "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
    log "done: installing NodeSource repository setup"
    log "installing nodejs"
    apt-get install -y -qq nodejs
    log "done: installing nodejs"
    log "installing OpenCode CLI"
    bash -lc "curl -fsSL https://opencode.ai/install | bash"
    log "done: installing OpenCode CLI"

    OPENCODE_BIN="$HOME/.opencode/bin/opencode"
    if [ ! -f "$OPENCODE_BIN" ]; then
        log "ERROR: expected OpenCode binary missing at ${OPENCODE_BIN}"
        find / -name opencode -type f 2>/dev/null || true
        exit 1
    fi

    if [ ! -f /tmp/upload/opencode_api_key.txt ]; then
        log "ERROR: /tmp/upload/opencode_api_key.txt not found"
        exit 1
    fi
    OPENCODE_API_KEY="$(cat /tmp/upload/opencode_api_key.txt)"

    log "starting OpenCode server on :4096"
    cd /workspace
    OPENCODE_API_KEY="$OPENCODE_API_KEY" \
    OPENCODE_CONFIG="/workspace/opencode.jsonc" \
        "$OPENCODE_BIN" serve --port 4096 --hostname 0.0.0.0 > /tmp/opencode.log 2>&1 &
    OPENCODE_PID=$!
    echo "$OPENCODE_PID" > /tmp/opencode.pid
    log "OpenCode process started (pid=${OPENCODE_PID})"
    wait_for_http "opencode" "http://localhost:4096/global/health" 120

    log "setup complete: envoi=:8000 opencode=:4096"
    exit 0
fi

if [ "$AGENT_KIND" = "codex" ]; then
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)
            TARGET_TRIPLE="x86_64-unknown-linux-musl"
            ;;
        aarch64|arm64)
            TARGET_TRIPLE="aarch64-unknown-linux-musl"
            ;;
        *)
            log "ERROR: unsupported architecture for Codex binary: $ARCH"
            exit 1
            ;;
    esac

    CODEX_TARBALL_URL="https://github.com/openai/codex/releases/latest/download/codex-${TARGET_TRIPLE}.tar.gz"
    tmpdir="$(mktemp -d)"
    log "downloading Codex CLI (${TARGET_TRIPLE})"
    curl -fsSL "$CODEX_TARBALL_URL" -o "$tmpdir/codex.tar.gz"
    log "done: downloading Codex CLI (${TARGET_TRIPLE})"
    log "extracting Codex CLI archive"
    tar -xzf "$tmpdir/codex.tar.gz" -C "$tmpdir"
    log "done: extracting Codex CLI archive"

    CODEX_EXTRACTED_BIN="$tmpdir/codex-${TARGET_TRIPLE}"
    if [ ! -f "$CODEX_EXTRACTED_BIN" ]; then
        log "ERROR: expected Codex binary not found at ${CODEX_EXTRACTED_BIN}"
        ls -la "$tmpdir"
        exit 1
    fi

    log "installing Codex CLI to /usr/local/bin"
    install -m 0755 "$CODEX_EXTRACTED_BIN" /usr/local/bin/codex
    log "done: installing Codex CLI to /usr/local/bin"
    if CODEX_VERSION="$(codex --version 2>/dev/null)"; then
        log "codex version: ${CODEX_VERSION}"
    fi
    mkdir -p /tmp/codex-home
    log "setup complete: envoi=:8000 codex=/usr/local/bin/codex"
    exit 0
fi

log "ERROR: unsupported AGENT_KIND=$AGENT_KIND"
exit 1
