"""
Main orchestrator for envoi-trace.

Uses the blocking POST /session/:id/message endpoint — one curl call per turn,
returns when the agent finishes its full cycle.

Usage:
    modal run orchestrate.py --max-turns 5 --model opencode/claude-sonnet-4-6
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import boto3
import modal
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

app = modal.App("envoi-trace")

DEFAULT_MODEL = "opencode/claude-sonnet-4"
TURN_TIMEOUT_SECONDS = 600  # 10 min per turn (agent writes files + builds + tests)


# ---------------------------------------------------------------------------
# Timestamped print
# ---------------------------------------------------------------------------


def ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def tprint(*args: Any, **kwargs: Any) -> None:
    if "flush" not in kwargs:
        kwargs["flush"] = True
    builtins.print(f"[{ts()}]", *args, **kwargs)


print = tprint

# ---------------------------------------------------------------------------
# Load files at import time (Modal serializes these into the function image)
# ---------------------------------------------------------------------------

PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text()
SETUP_SH = (Path(__file__).parent / "sandbox" / "setup.sh").read_text()
MCP_SERVER = (Path(__file__).parent / "sandbox" / "mcp_server.py").read_text()
OPENCODE_CONFIG = (Path(__file__).parent / "sandbox" / "opencode.jsonc").read_text()
ENVIRONMENT_DIR = Path(__file__).parent / "environment"

ENVIRONMENT_PY_FILES = {
    str(py_file.relative_to(ENVIRONMENT_DIR)): py_file.read_text()
    for py_file in ENVIRONMENT_DIR.rglob("*.py")
}
ENVIRONMENT_C_FILES = {
    str(c_file.relative_to(ENVIRONMENT_DIR)): c_file.read_text()
    for c_file in ENVIRONMENT_DIR.rglob("*.c")
}
ENVIRONMENT_TXT_FILES = {
    str(txt_file.relative_to(ENVIRONMENT_DIR)): txt_file.read_text()
    for txt_file in ENVIRONMENT_DIR.rglob("*.txt")
}

REQUIRED_PATHS: list[str] = [
    "basics",
    *[f"wacct/chapter_{i}" for i in range(1, 21)],
    *[f"c_testsuite/part_{i}" for i in range(1, 6)],
    *[f"torture/part_{i}" for i in range(1, 11)],
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def resolve_model(model: str) -> str:
    if "/" in model:
        return model
    return f"opencode/{model}"


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and any(
                token in key.lower() for token in ["key", "token", "secret", "password"]
            ):
                redacted[key] = "***" if val else val
            else:
                redacted[key] = redact_secrets(val)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def parse_http_status(output: str) -> tuple[str, int | None]:
    if "HTTP_STATUS:" not in output:
        return output, None
    body, status_part = output.rsplit("HTTP_STATUS:", 1)
    status_line = status_part.strip().splitlines()[0] if status_part.strip() else ""
    try:
        status = int(status_line)
    except ValueError:
        status = None
    return body.rstrip(), status


def model_to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    if hasattr(model, "json"):
        return model.json()
    if hasattr(model, "dict"):
        return json.dumps(model.dict())
    return json.dumps(model)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    passed: int
    failed: int
    total: int
    tests: list[dict[str, Any]] = Field(default_factory=list)


class EnvoiCall(BaseModel):
    path: str
    timestamp: str
    duration_ms: int
    status_code: int
    error: str | None = None
    result: TestResult | None = None


class SessionEnd(BaseModel):
    reason: Literal["solved", "turn_limit", "timeout", "agent_error", "envoi_error"]
    total_turns: int
    final_git_commit: str | None = None


class TurnRecord(BaseModel):
    trajectory_id: str
    session_id: str
    turn: int | None
    timestamp: str
    agent_model: str
    git_commit: str | None = None
    message_id: str | None = None
    envoi_calls: list[EnvoiCall] = Field(default_factory=list)
    session_end: SessionEnd | None = None


# ---------------------------------------------------------------------------
# Solve tracker
# ---------------------------------------------------------------------------


class SolveTracker:
    def __init__(self) -> None:
        self.solved: set[str] = set()
        self.all_calls: list[EnvoiCall] = []

    def update(self, envoi_calls: list[EnvoiCall]) -> None:
        self.all_calls.extend(envoi_calls)
        for call in envoi_calls:
            if call.result and call.result.total > 0 and call.result.passed == call.result.total:
                self.solved.add(call.path)

    def is_fully_solved(self) -> bool:
        return self.solved >= set(REQUIRED_PATHS)

    def get_unsolved_paths(self) -> list[str]:
        return [p for p in REQUIRED_PATHS if p not in self.solved]

    def get_latest_call_for_path(self, path: str) -> EnvoiCall | None:
        for call in reversed(self.all_calls):
            if call.path == path:
                return call
        return None


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


def summarize_tool_input(name: str, input_data: Any) -> str:
    if not isinstance(input_data, dict):
        return truncate_text(str(input_data), limit=200)
    if name == "bash":
        return truncate_text(input_data.get("command", ""), limit=200)
    if name == "read":
        return str(input_data.get("filePath") or input_data.get("path") or "?")
    if name in {"write", "edit"}:
        path = input_data.get("filePath") or input_data.get("path") or "?"
        content = input_data.get("content") or input_data.get("newString") or ""
        return f"{path} ({len(content)} bytes)"
    if name == "run_tests":
        return input_data.get("test_path", truncate_text(json.dumps(input_data), limit=200))
    return truncate_text(json.dumps(input_data), limit=200)


def log_message_parts(message: dict[str, Any]) -> None:
    """Print a human-readable summary of every part in a message."""
    info = message.get("info", {})
    role = info.get("role", "?")
    parts = message.get("parts", [])
    if not parts:
        return
    for part in parts:
        ptype = part.get("type", "?")
        if ptype == "text":
            text = part.get("text", "").strip()
            if text:
                print(f"  [{role}] {truncate_text(text, limit=300)}")
        elif ptype == "tool":
            name = part.get("tool", "?")
            state = part.get("state", {})
            status = state.get("status", "?")
            summary = summarize_tool_input(name, state.get("input", {}))
            output = state.get("output") or state.get("metadata", {}).get("output") or ""
            output_str = truncate_text(str(output), limit=200) if output else ""
            print(f"  [{role}] {name} ({status}) {summary}")
            if output_str and status == "completed":
                print(f"         -> {output_str}")
        elif ptype == "tool_use":
            name = part.get("name", "?")
            status = part.get("status", "?")
            summary = summarize_tool_input(name, part.get("input", {}))
            print(f"  [{role}] {name} ({status}) {summary}")
        elif ptype == "tool_result":
            content = str(part.get("content", ""))
            if content:
                print(f"         -> {truncate_text(content, limit=200)}")
        elif ptype == "patch":
            files = part.get("files", [])
            print(f"  [{role}] patch: {files}")
        elif ptype in {"step-start", "step-finish"}:
            pass  # skip noise
        else:
            print(f"  [{role}] {ptype}")


def extract_envoi_calls(message_parts: list[dict[str, Any]]) -> list[EnvoiCall]:
    """Extract envoi test calls from message parts."""
    calls: list[EnvoiCall] = []
    # Handle tool_use + tool_result pairs (older format)
    tool_results: dict[str, dict[str, Any]] = {}
    for part in message_parts:
        if part.get("type") == "tool_result":
            tool_results[part.get("tool_use_id", "")] = part
    for part in message_parts:
        if part.get("type") == "tool_use" and part.get("name") == "run_tests":
            tool_result = tool_results.get(part.get("id", ""))
            if tool_result:
                content = tool_result.get("content", "")
                if isinstance(content, str):
                    try:
                        calls.append(EnvoiCall(**json.loads(content)))
                    except (json.JSONDecodeError, Exception):
                        pass
        # Handle "tool" type parts (OpenCode's actual format)
        if part.get("type") == "tool" and part.get("tool") == "run_tests":
            state = part.get("state", {})
            if state.get("status") == "completed":
                output = state.get("output") or state.get("metadata", {}).get("output") or ""
                try:
                    data = json.loads(output) if isinstance(output, str) else output
                    calls.append(EnvoiCall(**data))
                except Exception:
                    pass
    return calls


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _s3_client


def get_bucket() -> str:
    return os.environ.get("AWS_S3_BUCKET", "envoi-trace-data")


def append_jsonl_record(trajectory_id: str, record: TurnRecord) -> None:
    s3 = get_s3_client()
    bucket = get_bucket()
    key = f"trajectories/{trajectory_id}/trajectory.jsonl"
    line = model_to_json(record) + "\n"
    try:
        existing = s3.get_object(Bucket=bucket, Key=key)
        existing_data = existing["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            existing_data = b""
        else:
            raise
    new_data = existing_data + line.encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=new_data)
    print(f"[s3] saved turn to s3://{bucket}/{key}")


def upload_file(trajectory_id: str, filename: str, data: bytes) -> str:
    s3 = get_s3_client()
    bucket = get_bucket()
    key = f"trajectories/{trajectory_id}/{filename}"
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


def save_messages_snapshot(trajectory_id: str, messages: list[dict[str, Any]]) -> None:
    """Save full message dump to S3 (overwrites same file each turn)."""
    try:
        s3 = get_s3_client()
        bucket = get_bucket()
        key = f"trajectories/{trajectory_id}/messages.json"
        body = json.dumps(messages, indent=2).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"[s3] saved messages ({len(messages)} msgs, {len(body)} bytes)")
    except Exception as e:
        print(f"[s3] failed to save messages: {e}")


# ---------------------------------------------------------------------------
# Modal images
# ---------------------------------------------------------------------------

function_image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "pydantic")
    .add_local_dir(Path(__file__).parent / "prompts", remote_path="/root/prompts")
    .add_local_dir(Path(__file__).parent / "sandbox", remote_path="/root/sandbox")
    .add_local_dir(Path(__file__).parent / "environment", remote_path="/root/environment")
)

sandbox_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install(
        "build-essential",
        "gcc",
        "g++",
        "clang",
        "git",
        "curl",
        "wget",
        "pkg-config",
        "libssl-dev",
    )
    .pip_install(
        "envoi @ git+https://github.com/TheSeamau5/envoi.git",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
        "mcp>=1.0.0",
    )
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
)


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


async def sandbox_run(
    sandbox: modal.Sandbox,
    cmd: str,
    timeout: int = 60,
    quiet: bool = False,
) -> tuple[int, str, str]:
    """Execute a command inside the sandbox."""
    if not quiet:
        print(f"[run] {cmd[:200]}")
    proc = await sandbox.exec.aio("bash", "-c", cmd)
    stdout, stderr = "", ""
    async for chunk in proc.stdout:
        stdout += chunk
    async for chunk in proc.stderr:
        stderr += chunk
    await proc.wait.aio()
    exit_code = proc.returncode or 0
    if exit_code != 0:
        print(f"[run] FAILED exit={exit_code} cmd={cmd[:100]}")
        if stderr:
            print(f"[run] stderr: {stderr[:500]}")
    return exit_code, stdout, stderr


async def sandbox_write_file(
    sandbox: modal.Sandbox,
    path: str,
    content: str,
    ensure_dir: bool = True,
) -> None:
    if ensure_dir:
        await sandbox_run(sandbox, f"mkdir -p '{Path(path).parent}'", quiet=True)
    async with await sandbox.open.aio(path, "w") as f:
        await f.write.aio(content)


# ---------------------------------------------------------------------------
# OpenCode API helpers — simple curl calls
# ---------------------------------------------------------------------------


async def create_session(sandbox: modal.Sandbox, title: str) -> str | None:
    payload = json.dumps({"title": title})
    await sandbox_write_file(sandbox, "/tmp/create_session.json", payload)
    _, stdout, _ = await sandbox_run(
        sandbox,
        "curl -sS -X POST http://localhost:4096/session "
        "-H 'Content-Type: application/json' -d @/tmp/create_session.json",
    )
    try:
        data = json.loads(stdout)
        session_id = data.get("id")
        print(f"[session] created id={session_id} slug={data.get('slug')}")
        return session_id
    except json.JSONDecodeError:
        print(f"[session] create failed: {stdout[:500]}")
        return None


async def stream_sse_events(
    sandbox: modal.Sandbox,
    stop_event: asyncio.Event,
) -> None:
    """
    Stream SSE events from GET /event for live visibility.
    Runs until stop_event is set.
    """
    proc = await sandbox.exec.aio(
        "bash",
        "-c",
        "curl -sS -N http://localhost:4096/event",
    )

    async def read_stream() -> None:
        buffer = ""
        async for chunk in proc.stdout:
            if stop_event.is_set():
                proc.terminate()
                return
            buffer += chunk
            while "\n\n" in buffer:
                event_data, buffer = buffer.split("\n\n", 1)
                if event_data.strip():
                    format_sse_event(event_data)

    try:
        await asyncio.wait_for(read_stream(), timeout=TURN_TIMEOUT_SECONDS + 30)
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        if not stop_event.is_set():
            print(f"[sse] error: {e}")
    finally:
        stop_event.set()
        try:
            proc.terminate()
        except Exception:
            pass


def format_sse_event(event_data: str) -> None:
    """Parse and pretty-print an SSE event."""
    lines = event_data.strip().split("\n")
    event_type = ""
    data = ""
    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()

    if not event_type or not data:
        return

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        print(f"[sse] {event_type}: {truncate_text(data, limit=100)}")
        return

    if event_type == "message.part.updated":
        part_type = payload.get("type", "?")
        if part_type == "text":
            text = payload.get("text", "")
            if text:
                print(f"[agent] {truncate_text(text, limit=200)}")
    elif event_type == "tool.called":
        tool_name = payload.get("name", "?")
        summary = summarize_tool_input(tool_name, payload.get("input", {}))
        print(f"[tool] {tool_name} → {summary}")
    elif event_type == "tool.completed":
        tool_name = payload.get("name", "?")
        status = payload.get("status", "?")
        output = payload.get("output", "")
        if tool_name == "run_tests":
            try:
                result = json.loads(output) if isinstance(output, str) else output
                passed = result.get("passed", "?")
                total = result.get("total", "?")
                print(f"[tool] {tool_name} done → {passed}/{total} tests")
            except Exception:
                print(f"[tool] {tool_name} done ({status})")
        else:
            print(f"[tool] {tool_name} done ({status}) → {truncate_text(str(output), limit=100)}")
    elif event_type == "session.status":
        status = payload.get("status", "?")
        print(f"[session] status={status}")
    elif event_type == "error":
        print(f"[sse] ERROR: {truncate_text(data, limit=200)}")


async def send_message_blocking(
    sandbox: modal.Sandbox,
    session_id: str,
    text: str,
    timeout: int = TURN_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """
    POST /session/:id/message — blocks until the agent finishes.
    Returns the full response with all parts (tool calls, text, etc.).
    Spawns SSE stream for live visibility while waiting.
    """
    payload = json.dumps({"parts": [{"type": "text", "text": text}]})
    await sandbox_write_file(sandbox, "/tmp/prompt.json", payload, ensure_dir=False)

    print(f"[prompt] sending message ({len(text)} chars), waiting up to {timeout}s...")

    stop_event = asyncio.Event()
    sse_task = asyncio.create_task(stream_sse_events(sandbox, stop_event))

    try:
        exit_code, stdout, stderr = await sandbox_run(
            sandbox,
            f"curl -sS -w '\nHTTP_STATUS:%{{http_code}}' -X POST "
            f"http://localhost:4096/session/{session_id}/message "
            f"-H 'Content-Type: application/json' -d @/tmp/prompt.json",
            timeout=timeout,
        )
    finally:
        stop_event.set()
        sse_task.cancel()
        try:
            await sse_task
        except asyncio.CancelledError:
            pass

    body, http_status = parse_http_status(stdout)
    print(f"[prompt] done exit={exit_code} http={http_status} body_len={len(body)}")

    if exit_code != 0 or (http_status is not None and http_status >= 400):
        print(f"[prompt] ERROR: {truncate_text(body, limit=1000)}")
        if stderr:
            print(f"[prompt] stderr: {stderr[:500]}")
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        print(f"[prompt] failed to parse response: {body[:500]}")
        return None


async def get_all_messages(sandbox: modal.Sandbox, session_id: str) -> list[dict[str, Any]]:
    """GET /session/:id/message — fetch all messages for the session."""
    _, stdout, _ = await sandbox_run(
        sandbox,
        f"curl -sS http://localhost:4096/session/{session_id}/message",
        quiet=True,
    )
    try:
        return json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        return []


async def get_git_commit(sandbox: modal.Sandbox) -> str | None:
    _, stdout, _ = await sandbox_run(
        sandbox,
        "cd /workspace && git rev-parse HEAD 2>/dev/null || echo none",
        quiet=True,
    )
    commit = stdout.strip()
    return commit[:16] if commit and commit != "none" else None


async def ensure_provider_connected(sandbox: modal.Sandbox, api_key: str) -> None:
    """Check if opencode provider is connected; if not, auth via PUT /auth/opencode."""
    _, stdout, _ = await sandbox_run(
        sandbox,
        "curl -sS http://localhost:4096/provider",
        quiet=True,
    )
    try:
        data = json.loads(stdout) if stdout.strip() else {}
        connected = data.get("connected", [])
        print(f"[provider] connected={connected}")
        if "opencode" in connected:
            return
    except json.JSONDecodeError:
        pass

    print("[provider] opencode not connected, setting auth...")
    payload = json.dumps({"apiKey": api_key})
    await sandbox_write_file(sandbox, "/tmp/auth.json", payload, ensure_dir=False)
    await sandbox_run(
        sandbox,
        "curl -sS -X PUT http://localhost:4096/auth/opencode "
        "-H 'Content-Type: application/json' -d @/tmp/auth.json",
    )


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------


async def setup_sandbox(sandbox: modal.Sandbox, model: str, api_key: str) -> None:
    print(f"[setup] model={model}")

    # Write files to sandbox
    await sandbox_write_file(sandbox, "/tmp/upload/setup.sh", SETUP_SH)
    await sandbox_write_file(sandbox, "/sandbox/mcp_server.py", MCP_SERVER)
    await sandbox_write_file(sandbox, "/tmp/upload/opencode_api_key.txt", api_key)

    config = OPENCODE_CONFIG.replace("MODEL_PLACEHOLDER", model)
    await sandbox_write_file(sandbox, "/workspace/opencode.jsonc", config)

    # Upload environment files (precreate dirs in one call)
    env_paths = (
        [f"/environment/{r}" for r in ENVIRONMENT_PY_FILES]
        + [f"/environment/{r}" for r in ENVIRONMENT_C_FILES]
        + [f"/environment/{r}" for r in ENVIRONMENT_TXT_FILES]
    )
    env_dirs = sorted({str(Path(p).parent) for p in env_paths})
    if env_dirs:
        await sandbox_run(sandbox, f"mkdir -p {' '.join(repr(d) for d in env_dirs)}", quiet=True)

    for rel, content in ENVIRONMENT_PY_FILES.items():
        await sandbox_write_file(sandbox, f"/environment/{rel}", content, ensure_dir=False)
    for rel, content in ENVIRONMENT_C_FILES.items():
        await sandbox_write_file(sandbox, f"/environment/{rel}", content, ensure_dir=False)
    for rel, content in ENVIRONMENT_TXT_FILES.items():
        await sandbox_write_file(sandbox, f"/environment/{rel}", content, ensure_dir=False)

    print(
        f"[setup] uploaded {len(ENVIRONMENT_PY_FILES)} py, "
        f"{len(ENVIRONMENT_C_FILES)} c, {len(ENVIRONMENT_TXT_FILES)} txt files"
    )

    # Run setup script (starts envoi + opencode)
    print("[setup] running setup.sh...")
    exit_code, stdout, stderr = await sandbox_run(
        sandbox,
        "bash /tmp/upload/setup.sh",
        timeout=600,
    )
    if exit_code != 0:
        print(f"[setup] FAILED:\n{stdout}\n{stderr}")
        raise RuntimeError(f"Setup failed (exit {exit_code})")
    print("[setup] done")


# ---------------------------------------------------------------------------
# End session
# ---------------------------------------------------------------------------


async def end_session(
    sandbox: modal.Sandbox,
    trajectory_id: str,
    session_id: str,
    turn_count: int,
    reason: Literal["solved", "turn_limit", "timeout", "agent_error", "envoi_error"],
    tracker: SolveTracker,
    last_message_id: str | None,
    model: str,
) -> None:
    print(f"[end] reason={reason} turns={turn_count}")

    final_commit = await get_git_commit(sandbox)

    end_record = TurnRecord(
        trajectory_id=trajectory_id,
        session_id=session_id,
        turn=None,
        timestamp=datetime.now(UTC).isoformat(),
        agent_model=model,
        git_commit=final_commit,
        message_id=None,
        envoi_calls=[],
        session_end=SessionEnd(
            reason=reason,
            total_turns=turn_count,
            final_git_commit=final_commit,
        ),
    )
    append_jsonl_record(trajectory_id, end_record)

    # Upload git bundle
    try:
        exit_code, _, _ = await sandbox_run(
            sandbox,
            "cd /workspace && git bundle create /tmp/repo.bundle --all",
            quiet=True,
        )
        _, size_out, _ = await sandbox_run(
            sandbox,
            "stat -c %s /tmp/repo.bundle 2>/dev/null || echo 0",
            quiet=True,
        )
        bundle_size = int(size_out.strip() or "0")
        print(f"[bundle] size={bundle_size} bytes")

        if bundle_size > 0:
            _, b64, _ = await sandbox_run(sandbox, "base64 /tmp/repo.bundle", quiet=True)
            data = base64.b64decode(b64.strip())
            upload_file(trajectory_id, "repo.bundle", data)
            print(f"[bundle] uploaded ({len(data)} bytes)")
    except Exception as e:
        print(f"[bundle] failed: {e}")

    print(f"[end] session ended: {reason}, {turn_count} turns, commit={final_commit}")


# ---------------------------------------------------------------------------
# Modal images
# ---------------------------------------------------------------------------


@app.function(
    timeout=14400,
    secrets=[modal.Secret.from_dotenv()],
    image=function_image,
)
async def run_trajectory(
    model: str = DEFAULT_MODEL,
    max_turns: int = 1000,
    timeout_seconds: int = 14400,
    trajectory_id: str | None = None,
) -> str:
    if trajectory_id is None:
        trajectory_id = str(uuid.uuid4())

    resolved_model = resolve_model(model)
    print(f"Starting trajectory {trajectory_id}")
    print(f"model={resolved_model} max_turns={max_turns} timeout={timeout_seconds}s")

    opencode_api_key = os.environ.get("OPENCODE_API_KEY", "")
    if not opencode_api_key:
        raise RuntimeError("OPENCODE_API_KEY not set")

    sandbox = await modal.Sandbox.create.aio(
        "bash",
        "-c",
        "sleep infinity",
        image=sandbox_image,
        timeout=timeout_seconds,
        app=app,
    )
    start_time = time.monotonic()

    try:
        # --- Setup ---
        await setup_sandbox(sandbox, resolved_model, opencode_api_key)

        session_id = await create_session(sandbox, title=f"trajectory-{trajectory_id}")
        if not session_id:
            raise RuntimeError("Failed to create OpenCode session")

        await ensure_provider_connected(sandbox, opencode_api_key)

        # --- Main loop: blocking message per turn ---
        tracker = SolveTracker()
        last_message_id: str | None = None
        turn_count = 0
        prompt_text = PROMPT
        end_reason: str | None = None

        try:
            while turn_count < max_turns:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout_seconds:
                    end_reason = "timeout"
                    break

                print(f"\n{'=' * 60}")
                print(f"TURN {turn_count + 1} / {max_turns}  (elapsed {int(elapsed)}s)")
                print(f"{'=' * 60}")

                # Send message and BLOCK until agent finishes
                response = await send_message_blocking(
                    sandbox,
                    session_id,
                    prompt_text,
                    timeout=TURN_TIMEOUT_SECONDS,
                )

                if response is None:
                    print("[turn] no response from agent")
                    end_reason = "agent_error"
                    break

                turn_count += 1

                # Log what the agent did
                info = response.get("info", {})
                parts = response.get("parts", [])
                last_message_id = info.get("id")
                print(f"[turn] response id={last_message_id} parts={len(parts)}")
                log_message_parts(response)

                # Also fetch ALL messages to see intermediate steps
                all_messages = await get_all_messages(sandbox, session_id)
                print(f"[turn] total session messages: {len(all_messages)}")

                # Log intermediate messages we haven't seen
                for msg in all_messages:
                    msg_role = msg.get("info", {}).get("role")
                    msg_id = msg.get("info", {}).get("id")
                    if msg_role == "assistant" and msg_id != last_message_id:
                        msg_parts = msg.get("parts", [])
                        if msg_parts:
                            print(f"  [intermediate msg {msg_id}]")
                            log_message_parts(msg)

                # Extract envoi calls from ALL messages
                all_envoi_calls: list[EnvoiCall] = []
                for msg in all_messages:
                    all_envoi_calls.extend(extract_envoi_calls(msg.get("parts", [])))

                tracker.update(all_envoi_calls)
                git_commit = await get_git_commit(sandbox)

                record = TurnRecord(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    turn=turn_count,
                    timestamp=datetime.now(UTC).isoformat(),
                    agent_model=resolved_model,
                    git_commit=git_commit,
                    message_id=last_message_id,
                    envoi_calls=all_envoi_calls,
                )
                append_jsonl_record(trajectory_id, record)
                save_messages_snapshot(trajectory_id, all_messages)

                solved_count = len(tracker.solved)
                total_count = len(REQUIRED_PATHS)
                print(
                    f"[turn] turn={turn_count} commit={git_commit} "
                    f"envoi_calls={len(all_envoi_calls)} solved={solved_count}/{total_count}"
                )

                if tracker.is_fully_solved():
                    end_reason = "solved"
                    break

                # Build re-injection prompt for next turn
                unsolved = tracker.get_unsolved_paths()
                details: list[str] = []
                for p in unsolved[:10]:
                    call = tracker.get_latest_call_for_path(p)
                    if call and call.result:
                        details.append(f"  - {p}: {call.result.passed}/{call.result.total}")
                    else:
                        details.append(f"  - {p}: not run")

                prompt_text = "Continue working on the compiler. Run tests and pass ALL suites."
                if details:
                    prompt_text += "\n\nCurrent test status:\n" + "\n".join(details)

            if end_reason is None:
                end_reason = "turn_limit"

        except Exception as loop_err:
            print(f"[error] crash during main loop: {loop_err}")
            end_reason = "agent_error"
            # Save whatever messages we have
            try:
                crash_messages = await get_all_messages(sandbox, session_id)
                if crash_messages:
                    save_messages_snapshot(trajectory_id, crash_messages)
                    print(f"[error] saved {len(crash_messages)} messages before crash")
            except Exception:
                print("[error] could not save crash messages")

        # Always end the session and save final state
        await end_session(
            sandbox,
            trajectory_id,
            session_id,
            turn_count,
            end_reason,
            tracker,
            last_message_id,
            resolved_model,
        )
        return trajectory_id

    except Exception as e:
        print(f"[error] {e}")
        raise
    finally:
        try:
            await sandbox.terminate.aio()
        except Exception:
            pass

    return trajectory_id


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
async def main(
    model: str = DEFAULT_MODEL,
    max_turns: int = 1000,
    trajectory_id: str | None = None,
) -> None:
    result = await run_trajectory.remote.aio(
        model=model,
        max_turns=max_turns,
        trajectory_id=trajectory_id,
    )
    print(f"Completed trajectory: {result}")
