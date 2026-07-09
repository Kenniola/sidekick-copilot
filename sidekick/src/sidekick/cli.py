"""Sidekick CLI — init, serve, and list-configs commands.

Entry points:
  sidekick init          — scaffold ~/.sidekick/ and register MCP in VS Code
  sidekick serve         — run the MCP server (used by mcp.json)
  sidekick list-configs  — show available customer profiles
"""

from __future__ import annotations

import importlib.resources
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so glyphs (✓, ⚠, —) never crash the CLI.

    On Windows the console/pipe encoding is often cp1252 (or the OEM code
    page); printing a checkmark then raises ``UnicodeEncodeError`` and aborts
    commands like ``init``/``uninstall`` mid-run — especially when output is
    piped. Reconfiguring to UTF-8 with ``errors="replace"`` is a safe no-op
    where it's already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # not a reconfigurable text stream (e.g. redirected buffer)


def _get_user_dir() -> Path:
    """Return ~/.sidekick/."""
    import os
    return Path(os.environ.get("SIDEKICK_HOME", Path.home() / ".sidekick"))


_KNOWN_STT_MODELS = {
    "tiny.en", "base.en", "small.en", "medium.en",
    "distil-large-v2", "distil-large-v3", "large-v3", "turbo", "large-v3-turbo",
}


def _set_env_stt_model(env_file: Path, model: str) -> None:
    """Ensure a single active ``SIDEKICK_WHISPER_MODEL=<model>`` line in the .env.

    Replaces any existing active *or commented* model line and de-duplicates,
    so repeated ``--stt-model`` runs stay idempotent.
    """
    key = "SIDEKICK_WHISPER_MODEL"
    new_line = f"{key}={model}"
    lines = (
        env_file.read_text(encoding="utf-8").splitlines()
        if env_file.exists()
        else []
    )
    out: list[str] = []
    replaced = False
    for ln in lines:
        stripped = ln.lstrip("# ").strip()
        if stripped.startswith(key + "="):
            if not replaced:
                out.append(new_line)
                replaced = True
            # else: drop duplicate/commented variants
        else:
            out.append(ln)
    if not replaced:
        out.append(new_line)
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")


def _apply_stt_model(env_file: Path, model: str) -> None:
    """Validate and persist the chosen Whisper model into ~/.sidekick/.env."""
    if model not in _KNOWN_STT_MODELS:
        print(
            f"\u26a0\ufe0f  '{model}' is not a recognised Whisper model - setting it anyway."
        )
    _set_env_stt_model(env_file, model)
    print(f"\u2713 Set Whisper model to '{model}' in {env_file}")
    print("   (run `sidekick benchmark-stt` to confirm it runs in real time)")


def _get_vscode_user_settings_path() -> Path | None:
    """Find the VS Code User settings directory (cross-platform)."""
    import platform
    system = platform.system()
    if system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming" / "Code" / "User"
    elif system == "Darwin":
        appdata = Path.home() / "Library" / "Application Support" / "Code" / "User"
    else:
        appdata = Path.home() / ".config" / "Code" / "User"
    return appdata if appdata.exists() else None


def _cmd_init():
    """Scaffold ~/.sidekick/ and register the MCP server in VS Code."""
    user_dir = _get_user_dir()
    print(f"Initialising Sidekick at {user_dir}\n")

    # 1. Create directory structure
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "cache").mkdir(exist_ok=True)
    (user_dir / "outputs").mkdir(exist_ok=True)
    print(f"\u2713 Created {user_dir}/")

    # 1b. Platform checks
    _check_platform()

    # 2. Copy customers.yaml starter (if not exists)
    customers_file = user_dir / "customers.yaml"
    if not customers_file.exists():
        try:
            template_ref = importlib.resources.files("sidekick") / "configs" / "_template.yaml"
            template_content = template_ref.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError):
            template_content = (
                "# Sidekick — Customer Profiles\n"
                "# Each top-level key is a profile name.\n"
                "# Usage: @sidekick listen --config <profile>\n\n"
                "# example:\n"
                "#   customer: Example Corp\n"
                "#   description: \"Your engagement description\"\n"
                "#   participants:\n"
                "#     consultant: [\"Your Name\"]\n"
            )
        customers_file.write_text(template_content, encoding="utf-8")
        print(f"\u2713 Created {customers_file}")
    else:
        print(f"\u2713 {customers_file} already exists (kept)")

    # 2b. Create .env file for secrets (if not exists)
    env_file = user_dir / ".env"
    if not env_file.exists():
        env_content = (
            "# Sidekick secrets — loaded automatically at startup.\n"
            "# This file is local to ~/.sidekick/ and never committed to any repo.\n"
            "\n"
            "# GitHub token (only needed if `gh auth token` is unavailable)\n"
            "# GITHUB_TOKEN=ghp_xxx\n"
            "\n"
            "# Whisper model override (optional). Run `sidekick benchmark-stt`\n"
            "# to measure which model your machine can run in real time, then\n"
            "# `sidekick init --stt-model <model>` to persist the choice here.\n"
            "# Default is small.en (~470MB, ~5-7% WER). Other choices:\n"
            "#   base.en          ~150MB  fastest, ~8-10% WER\n"
            "#   medium.en        ~1.5GB  better English accuracy\n"
            "#   distil-large-v3  ~1.5GB  RECOMMENDED - near large-v3 accuracy, low hallucination\n"
            "#   large-v3         ~3.1GB  best, GPU recommended\n"
            "# SIDEKICK_WHISPER_MODEL=small.en\n"
            "# SIDEKICK_WHISPER_COMPUTE=int8\n"
            "\n"
            "# Web search grounding (optional).\n"
            "# research always uses the free Microsoft Learn API (no key).\n"
            "# Set ONE key below to also pull verified non-Microsoft sources\n"
            "# (AWS, Databricks, Spark, PostgreSQL). Tavily is preferred when\n"
            "# both are present. No key = Microsoft Learn only (no error).\n"
            "#   Tavily: https://tavily.com   Brave: https://brave.com/search/api/\n"
            "# TAVILY_API_KEY=tvly-xxxxxxxx\n"
            "# BRAVE_API_KEY=BSA-xxxxxxxx\n"
        )
        env_file.write_text(env_content, encoding="utf-8")
        print(f"\u2713 Created {env_file}")
    else:
        print(f"\u2713 {env_file} already exists (kept)")

    # 2c. Optional STT model selection: `sidekick init --stt-model <model>`
    stt_model = _arg_value("--stt-model")
    if stt_model:
        _apply_stt_model(env_file, stt_model)

    # 3. Check for GitHub token
    gh_token = _check_github_token()

    # 4. Register MCP server in VS Code User Settings
    _register_mcp_server()

    # 5. Install sidekick-notify VS Code extension
    _install_notify_extension()

    # 6. Deploy agent instructions to VS Code User prompts
    _install_agent_definition()

    # 7. Summary
    print("\n" + "\u2501" * 50)
    print("\u2713 Sidekick is ready!\n")
    if not gh_token:
        print("\u26a0\ufe0f  No GitHub token detected.")
        print("   Install gh CLI and run: gh auth login")
        print("   Or set GITHUB_TOKEN in your environment.\n")
    print("Next steps:")
    print(f"  1. Edit your customer profiles: {customers_file}")
    print("  2. In VS Code Copilot Chat, type:")
    print("       @sidekick listen              \u2014 start with defaults")
    print("       @sidekick listen --config acme \u2014 use a customer profile")
    print(f"\nConfig reference: {user_dir / 'customers.yaml'}")


def _check_platform():
    """Warn about platform-specific limitations."""
    import platform
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        # Under x64 Python emulation (our default on ARM64), this block
        # won't trigger because platform.machine() returns 'amd64'.
        # It only fires if someone installs with native ARM64 Python.
        print("\u26a0\ufe0f  ARM64 detected with native ARM64 Python.")
        print("   Install with x64 Python for full compatibility (the installer does this automatically).")


def _check_github_token() -> bool:
    """Check if a GitHub token is available."""
    import os
    if os.environ.get("GITHUB_TOKEN"):
        print("\u2713 GitHub token found (GITHUB_TOKEN env var)")
        return True

    # Try gh CLI
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("\u2713 GitHub token available via gh CLI")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def _install_notify_extension():
    """Install the sidekick-notify VS Code extension from the bundled vsix."""
    try:
        vsix_ref = importlib.resources.files("sidekick") / "extensions" / "sidekick-notify.vsix"
        vsix_path = str(vsix_ref)
    except (FileNotFoundError, TypeError):
        print("\u26a0\ufe0f  sidekick-notify.vsix not found in package — skipping extension install")
        return

    # Check if VS Code CLI is available
    code_cmd = shutil.which("code")
    if not code_cmd:
        print("\u26a0\ufe0f  'code' CLI not found — install the extension manually:")
        print(f"   code --install-extension {vsix_path}")
        return

    try:
        result = subprocess.run(
            [code_cmd, "--install-extension", vsix_path, "--force"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("\u2713 Installed sidekick-notify VS Code extension")
        else:
            # Extension may already be installed or VS Code returned a warning
            if "already installed" in (result.stdout + result.stderr).lower():
                print("\u2713 sidekick-notify extension already installed")
            else:
                print(f"\u26a0\ufe0f  Extension install returned: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, OSError):
        print("\u26a0\ufe0f  Could not install extension automatically")
        print(f"   Run manually: code --install-extension {vsix_path}")


def _get_vscode_prompts_path() -> Path | None:
    """Return the VS Code User prompts directory."""
    vscode_dir = _get_vscode_user_settings_path()
    if not vscode_dir:
        return None
    return vscode_dir / "prompts"


def _install_agent_definition():
    """Deploy sidekick.agent.md to the VS Code User prompts folder."""
    prompts_dir = _get_vscode_prompts_path()
    if not prompts_dir:
        print("\u26a0\ufe0f  VS Code User settings not found — skipping agent definition install")
        return

    prompts_dir.mkdir(parents=True, exist_ok=True)
    agent_dest = prompts_dir / "sidekick.agent.md"

    try:
        agent_ref = importlib.resources.files("sidekick") / "agents" / "sidekick.agent.md"
        agent_content = agent_ref.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        print("\u26a0\ufe0f  sidekick.agent.md not found in package — skipping agent definition")
        return

    agent_dest.write_text(agent_content, encoding="utf-8")
    print(f"\u2713 Installed agent definition at {agent_dest}")


def _uninstall_agent_definition():
    """Remove sidekick.agent.md from the VS Code User prompts folder."""
    prompts_dir = _get_vscode_prompts_path()
    if not prompts_dir:
        return

    agent_file = prompts_dir / "sidekick.agent.md"
    if agent_file.exists():
        agent_file.unlink()
        print(f"\u2713 Removed agent definition from {agent_file}")
    else:
        print("\u2713 Agent definition not found (already removed)")


def _register_mcp_server():
    """Register sidekick as an MCP server in VS Code User Settings."""
    vscode_dir = _get_vscode_user_settings_path()
    if not vscode_dir:
        print("\u26a0\ufe0f  VS Code User settings directory not found — skipping MCP registration")
        print("   Add the MCP config manually to .vscode/mcp.json in your workspace")
        return

    mcp_file = vscode_dir / "mcp.json"

    # Build the MCP server entry
    python_path = sys.executable
    server_entry = {
        "command": "powershell",
        "args": [
            "-NoProfile",
            "-Command",
            f"$env:GITHUB_TOKEN = (gh auth token); & '{python_path}' -m sidekick.server",
        ],
        # Point grounding/research at the user's currently open workspace. Both
        # build_grounding_context() and the research pipeline read
        # SIDEKICK_WORKSPACE_ROOT (default "."); without this they resolve
        # relative to the MCP server's process cwd, which is unreliable and
        # silently skips the team's .github/instructions standards. VS Code
        # substitutes ${workspaceFolder} when it launches the server.
        "env": {
            "SIDEKICK_WORKSPACE_ROOT": "${workspaceFolder}",
        },
    }

    # Load or create mcp.json
    mcp_config: dict = {"servers": {}}
    if mcp_file.exists():
        try:
            mcp_config = json.loads(mcp_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            # Backup corrupt file
            backup = mcp_file.with_suffix(".json.bak")
            shutil.copy2(mcp_file, backup)
            print(f"\u26a0\ufe0f  Backed up corrupt mcp.json to {backup}")
            mcp_config = {"servers": {}}

    if "servers" not in mcp_config:
        mcp_config["servers"] = {}

    if "sidekick" in mcp_config["servers"]:
        print(f"\u2713 MCP server already registered in {mcp_file}")
        return

    mcp_config["servers"]["sidekick"] = server_entry
    mcp_file.write_text(
        json.dumps(mcp_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\u2713 Registered MCP server in {mcp_file}")


def _cmd_serve():
    """Run the MCP server (called by mcp.json)."""
    import asyncio
    import logging
    from sidekick.server import server

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    asyncio.run(server.run_stdio_async())


def _cmd_list_configs():
    """List available customer profiles."""
    from sidekick.config import list_available_configs, get_user_dir

    profiles = list_available_configs()
    user_dir = get_user_dir()

    if not profiles:
        print("No customer profiles found.")
        print(f"Add profiles to: {user_dir / 'customers.yaml'}")
        return

    print(f"Available profiles ({user_dir}):\n")
    for name in profiles:
        print(f"  \u2022 {name}")
    print("\nUsage: @sidekick listen --config <profile>")


def _cmd_models():
    """Print the resolved per-tier model fallback chains.

    Usage: sidekick models [profile]

    Shows the chain call_llm uses for each tier after applying YAML config
    and any SIDEKICK_MODEL_<TIER> environment overrides.
    """
    import os
    from sidekick.config import load_config

    profile = "default"
    # First positional arg after "models" that isn't a flag
    for a in sys.argv[2:]:
        if not a.startswith("-"):
            profile = a
            break

    try:
        config = load_config(profile)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(f"Resolved model chains (profile: {profile})\n")
    for tier in ("fast", "standard", "deep"):
        env_key = f"SIDEKICK_MODEL_{tier.upper()}"
        overridden = bool(os.environ.get(env_key, "").strip())
        chain = config.models.chain(tier)
        suffix = f"  [overridden by {env_key}]" if overridden else ""
        print(f"  {tier:9}{suffix}")
        for i, (provider, model) in enumerate(chain):
            marker = "primary " if i == 0 else "fallback"
            print(f"    {marker}  {provider}:{model}")
        print()
    print("Override a tier without editing YAML, e.g.:")
    print('  $env:SIDEKICK_MODEL_DEEP = "copilot:claude-opus-4.8,copilot:gpt-4.1"')


def _arg_value(flag: str) -> str | None:
    """Return the value following ``flag`` in argv, or None if absent.

    Supports ``--flag value`` and ``--flag=value``.
    """
    for i, a in enumerate(sys.argv):
        if a == flag:
            return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _cmd_benchmark_stt():
    """Benchmark Whisper models' real-time factor (RTF) on this machine.

    Usage:
      sidekick benchmark-stt [--audio PATH] [--seconds N]
                             [--models a,b,c] [--threshold F]

    With no ``--audio`` it records ~N seconds of system audio (loopback) — play
    a meeting recording or talk while it captures. Prints an RTF table and
    recommends the most accurate model that stays under the threshold.
    """
    from sidekick.transcript import benchmark as bm

    audio_path = _arg_value("--audio")
    seconds = int(_arg_value("--seconds") or 20)
    threshold = float(_arg_value("--threshold") or bm.DEFAULT_RTF_THRESHOLD)
    models_arg = _arg_value("--models")
    candidates = (
        [m.strip() for m in models_arg.split(",") if m.strip()]
        if models_arg
        else bm.CANDIDATE_MODELS
    )

    if audio_path:
        try:
            audio, duration = bm.load_audio(audio_path)
        except Exception as e:  # noqa: BLE001 — surface a clear CLI error
            print(f"Could not load audio {audio_path}: {type(e).__name__}: {e}")
            sys.exit(1)
    else:
        print(f"No --audio given; recording ~{seconds}s of system audio (loopback).")
        print("Play a meeting recording or talk now...\n")
        try:
            audio, duration = bm.record_loopback(seconds)
        except Exception as e:  # noqa: BLE001 — guide the user to --audio
            print(f"Recording failed ({type(e).__name__}: {e}).")
            print("Pass a file instead:  sidekick benchmark-stt --audio PATH.wav")
            sys.exit(1)

    if duration < 1.0:
        print(f"Audio too short ({duration:.1f}s) — need ~1s+ of speech.")
        sys.exit(1)

    print(f"Benchmarking {len(candidates)} model(s) on {duration:.1f}s of audio...")
    print("(first run downloads weights — subsequent runs are cached)\n")
    results = bm.run_benchmark(audio, duration, candidates, bm.default_transcribe_fn)

    print(f"{'model':<20}{'load s':>8}{'decode s':>10}{'RTF':>7}  status")
    print("-" * 55)
    for r in results:
        if r.ok:
            status = "ok" if r.rtf < threshold else "SLOW"
            print(
                f"{r.model:<20}{r.load_seconds:>8.1f}"
                f"{r.transcribe_seconds:>10.1f}{r.rtf:>7.2f}  {status}"
            )
        else:
            print(f"{r.model:<20}{'--':>8}{'--':>10}{'--':>7}  FAILED: {r.error}")

    print("\nTranscript previews (eyeball accuracy):")
    for r in results:
        if r.ok and r.sample_text:
            print(f"  [{r.model}] {r.sample_text}")

    rec = bm.recommend_model(results, threshold)
    print()
    if rec:
        print(f"Recommended (most accurate with RTF < {threshold}): {rec}")
        print(f"Apply it:  sidekick init --stt-model {rec}")
        print(f"       or:  set SIDEKICK_WHISPER_MODEL={rec}")
    else:
        print(
            f"No model met RTF < {threshold} on this machine. Use the fastest "
            "that works (small.en/base.en), or run on a CUDA GPU."
        )


def _running_inside_uv_tool() -> bool:
    """True when the current interpreter lives inside the uv tool environment.

    When ``sidekick uninstall`` is launched via the installed ``sidekick.exe``,
    ``sys.executable`` is the Python interpreter *inside*
    ``…/uv/tools/sidekick-copilot``. uv cannot delete that tree while this
    process holds files in it (Windows file lock), so we must defer the removal
    until after we exit.
    """
    exe = str(Path(sys.executable).resolve()).lower().replace("\\", "/")
    return "/uv/tools/sidekick-copilot" in exe


def _uninstall_uv_tool() -> None:
    """Remove the ``sidekick-copilot`` uv tool environment.

    Historically this ran ``uv tool uninstall`` directly from inside the tool's
    own environment, which Windows file-locks — the failure was swallowed and
    reported as the misleading "not in uv tools (already removed)", leaving a
    corrupted ``%APPDATA%\\uv\\tools\\sidekick-copilot`` behind. When we detect
    that self-lock we hand the uninstall to a detached helper that waits for
    this process to exit first; otherwise we remove it synchronously with
    honest messaging.
    """
    import os

    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        print(
            "\u26a0\ufe0f  uv not found — if installed via pip, run: pip uninstall sidekick-copilot"
        )
        return

    if _running_inside_uv_tool() and sys.platform == "win32":
        # Spawn a detached PowerShell that waits for *this* PID to exit, then
        # runs the uninstall once the file lock is released.
        parent_pid = os.getpid()
        ps_script = (
            f"$ppid = {parent_pid}; "
            "while (Get-Process -Id $ppid -ErrorAction SilentlyContinue) "
            "{ Start-Sleep -Milliseconds 300 }; "
            f"& '{uv_cmd}' tool uninstall sidekick-copilot | Out-Null"
        )
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        try:
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    ps_script,
                ],
                creationflags=(
                    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                ),
                close_fds=True,
            )
            print(
                "\u2713 Scheduled sidekick-copilot uv tool removal "
                "(completes a moment after this process exits)"
            )
        except OSError:
            print("\u26a0\ufe0f  Could not schedule uv tool removal — once this exits, run:")
            print("   uv tool uninstall sidekick-copilot")
        return

    # Not self-locked — uninstall now and report the real outcome.
    try:
        result = subprocess.run(
            [uv_cmd, "tool", "uninstall", "sidekick-copilot"],
            capture_output=True, text=True, timeout=30,
        )
        combined = (result.stdout + result.stderr).lower()
        if result.returncode == 0:
            print("\u2713 Removed sidekick-copilot uv tool environment")
        elif "not installed" in combined or "no tool" in combined or "not found" in combined:
            print("\u2713 sidekick-copilot not in uv tools (already removed)")
        else:
            print("\u26a0\ufe0f  uv tool uninstall failed — run manually:")
            print("   uv tool uninstall sidekick-copilot")
            tail = (result.stderr or result.stdout).strip().splitlines()
            if tail:
                print(f"   ({tail[-1]})")
    except (subprocess.TimeoutExpired, OSError):
        print("\u26a0\ufe0f  Could not remove uv tool — run manually:")
        print("   uv tool uninstall sidekick-copilot")


def _cmd_uninstall():
    """Remove all sidekick artifacts from the system."""
    user_dir = _get_user_dir()

    print("Sidekick Uninstaller")
    print("=" * 40)
    print()
    print("This will remove:")
    print(f"  1. {user_dir}/ (config, cache, outputs, session logs)")
    print("  2. MCP server entry from VS Code User settings")
    print("  3. sidekick-notify VS Code extension")
    print("  4. sidekick agent definition from VS Code User prompts")
    print("  5. sidekick-copilot uv tool environment")
    print()

    # Check --yes flag for non-interactive use
    skip_confirm = "--yes" in sys.argv or "-y" in sys.argv

    if not skip_confirm:
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    print()

    # 1. Remove MCP entry from VS Code User settings
    _unregister_mcp_server()

    # 2. Uninstall sidekick-notify extension
    code_cmd = shutil.which("code")
    if code_cmd:
        try:
            result = subprocess.run(
                [code_cmd, "--uninstall-extension", "koladimeji.sidekick-notify"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print("\u2713 Uninstalled sidekick-notify extension")
            else:
                print("\u2713 sidekick-notify extension not found (already removed)")
        except (subprocess.TimeoutExpired, OSError):
            print("\u26a0\ufe0f  Could not uninstall extension — run manually:")
            print("   code --uninstall-extension koladimeji.sidekick-notify")
    else:
        print("\u26a0\ufe0f  VS Code CLI not found — uninstall extension manually if installed")

    # 3. Remove agent definition from VS Code User prompts
    _uninstall_agent_definition()

    # 4. Remove ~/.sidekick/ directory
    if user_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(user_dir, ignore_errors=True)
        if not user_dir.exists():
            print(f"\u2713 Removed {user_dir}/")
        else:
            print(f"\u26a0\ufe0f  Could not fully remove {user_dir}/ — delete manually")
    else:
        print(f"\u2713 {user_dir}/ not found (already removed)")

    # 5. Remove uv tool environment
    _uninstall_uv_tool()

    print()
    print("\u2501" * 40)
    print("\u2713 Sidekick uninstalled.")
    print()
    print("Not removed (shared tools, remove manually if desired):")
    print("  - uv:  irm https://astral.sh/uv/uninstall.ps1 | iex")
    print("  - gh:  winget uninstall GitHub.cli")
    print()


def _unregister_mcp_server():
    """Remove sidekick entry from VS Code User mcp.json."""
    vscode_dir = _get_vscode_user_settings_path()
    if not vscode_dir:
        print("\u2713 VS Code settings not found (nothing to remove)")
        return

    mcp_file = vscode_dir / "mcp.json"
    if not mcp_file.exists():
        print("\u2713 No mcp.json found (nothing to remove)")
        return

    try:
        mcp_config = json.loads(mcp_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        print(f"\u26a0\ufe0f  Could not parse {mcp_file} — remove sidekick entry manually")
        return

    servers = mcp_config.get("servers", {})
    if "sidekick" not in servers:
        print("\u2713 No sidekick entry in mcp.json (already removed)")
        return

    del servers["sidekick"]
    mcp_file.write_text(json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8")
    print(f"\u2713 Removed sidekick from {mcp_file}")


def main():
    """CLI entry point: sidekick <command>."""
    _force_utf8_output()
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print("Sidekick \u2014 Real-time meeting co-pilot\n")
        print("Commands:")
        print("  sidekick init          Scaffold ~/.sidekick/ and register in VS Code")
        print("  sidekick serve         Run the MCP server (used by mcp.json)")
        print("  sidekick list-configs  Show available customer profiles")
        print("  sidekick models        Show resolved per-tier model chains")
        print("  sidekick benchmark-stt Benchmark Whisper models' speed on this machine")
        print("  sidekick uninstall     Remove all sidekick artifacts")
        print("  sidekick help          Show this help message")
        return

    command = args[0]
    if command == "init":
        _cmd_init()
    elif command == "serve":
        _cmd_serve()
    elif command == "list-configs":
        _cmd_list_configs()
    elif command == "models":
        _cmd_models()
    elif command == "benchmark-stt":
        _cmd_benchmark_stt()
    elif command == "uninstall":
        _cmd_uninstall()
    else:
        print(f"Unknown command: {command}")
        print("Run 'sidekick help' for available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
