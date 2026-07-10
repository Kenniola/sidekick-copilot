# Sidekick — Installation Guide

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows x64 | Fully supported | Native install, all features |
| Windows ARM64 | Fully supported | Installer auto-selects x64 Python; live audio runs via Windows x64 emulation |
| macOS / Linux | Not supported | Live audio capture uses WASAPI loopback (Windows-only API) |

## Prerequisites

- **Windows 10 or 11** (x64 or ARM64)
- **VS Code** with the **GitHub Copilot Chat** extension
- A **GitHub account** (`gh auth login`)
- **GitHub Copilot** recommended — **any plan** (Individual/Pro/Business/Enterprise)

That's it. Python, GitHub CLI, and Azure resources are all installed or handled automatically.

> **On the model backend:** Sidekick calls the **GitHub Copilot API** (Claude +
> GPT models) using your `gh` token — this needs an active Copilot plan, *not*
> specifically Enterprise. If you have **no Copilot plan**, calls automatically
> fall back to the **free GitHub Models API** (any GitHub account); it works but
> is rate-limited and lower quality for a busy live meeting. No key or config is
> needed either way.


---

## Install

### Step 1 — Open PowerShell

Press `Win + X` → select **Terminal** (or **Windows PowerShell**).
Alternatively, press `Win + R`, type `powershell`, press Enter.

### Step 2 — Paste this single line and press Enter

```powershell
irm https://raw.githubusercontent.com/Kenniola/sidekick-copilot/main/sidekick/install.ps1 | iex
```

The installer will:
- Install **uv** (Python package manager) and **Python** — if not already present
- Install **GitHub CLI** — if not already present (you'll be asked to run `gh auth login`, then re-run the line above)
- Install **sidekick-copilot** and all dependencies
- Create your config at `~/.sidekick/`
- Register the MCP server in VS Code
- Install the notification extension

### Step 3 — Edit your profile

Open the file `%USERPROFILE%\.sidekick\customers.yaml` in any text editor and add your name:

```yaml
myproject:
  customer: Acme Corp
  consultant: Your Name
  description: "Data platform migration"
```

### Step 4 — Use it

Open **VS Code** → open **Copilot Chat** (`Ctrl+Shift+I`) → type:

```
@sidekick listen
@sidekick listen --config myproject
```

---

## Configuration

Everything lives in `~/.sidekick/customers.yaml` (on Windows, `~/.sidekick/` =
`C:\Users\<you>\.sidekick\`). Each top-level key is a **profile** (select with
`--config <name>`). Profiles **inherit** the bundled `default.yaml` and override
only what you set — so most profiles are 3–4 lines.

### What you actually need to set

```yaml
myproject:
  customer: Acme Corp                 # shown in summaries / deliverables
  consultant: Your Name               # your name(s) — used for speaker attribution
  description: "Fabric migration"     # one line; seeds context + STT vocabulary
```

Everything else is optional. The two biggest accuracy wins to add next:

```yaml
  glossary: [OneLake, Contoso Analytics]   # proper nouns to transcribe correctly
  domains: [Microsoft Fabric, PostgreSQL]  # bias vocabulary + research to your field
```

### Full toggle reference

Grouped by block. All are optional; the value shown is the **default**.

**Top level**

| Key | Default | What it does |
|-----|---------|--------------|
| `customer` | `General` | Name used in summaries, deliverables, and the output folder. |
| `consultant` | – | Your name(s). Roster for post-call speaker naming. |
| `client` | `["*"]` | Client attendee names — improves speaker attribution. |
| `description` | – | One-line engagement summary; seeds context + STT vocabulary. |
| `domains` | Fabric, Power BI, Azure Data Platform | **Soft prior** — biases STT vocabulary and research/analyst context. Not a filter; Sidekick still discusses anything. Override for your field. |
| `glossary` | – | Proper nouns / product / project names seeded into the Whisper prior so they're transcribed correctly from the first word. Strongest single accuracy lever. |
| `stt_corrections` | – | `"misheard": "correct"` map for stubborn homophones a glossary can't fix. |
| `objectives` | auto-inferred | Goals the relevance adjudicator scores against. Left empty, inferred from the opening minutes. |
| `rules` | 3 defaults | Guidance appended to the analyst / research prompts. |

**`sensitivity`** — how much Sidekick surfaces

| Key | Default | What it does |
|-----|---------|--------------|
| `accuracy_mode` | `true` | Two-stage pipeline: a fast detector proposes candidates, a periodic deep-tier adjudicator surfaces only the few most relevant. Higher precision, slight latency. `false` = older high-recall (noisier) mode. |
| `auto_suggest` | `true` | Occasionally surfaces one high-impact question to ask the client (an `[ask]` card). |
| `auto_suggest_interval_seconds` | `120` | Minimum gap between auto-suggestions. |
| `adjudicator_interval_seconds` | `40` | How often the deep adjudicator runs (lower = snappier, more model calls). |
| `adjudicator_pause_flush` | `true` | Surface an urgent hedge immediately without waiting for the interval. |
| `max_surfaced_per_pass` | `3` | Hard cap on findings surfaced per adjudicator pass. |
| `surface_threshold` | `0.7` | Precision gate (0–1); higher = stricter, fewer findings. |
| `answer_tier` | `auto` | `deep` forces deep-model answers everywhere (accuracy_mode already implies deep). |
| `self_critique` | `false` | Draft → critique → refine each answer. Higher accuracy, ~2× latency. |
| `analyst_interval_seconds` | `10` | How often the transcript is classified. |
| `verify_consultant_answers` | `true` | Fact-check statements the consultant makes. |
| `show_verifications` | `corrections_only` | `all` / `corrections_only` / `none`. |
| `trigger_threshold` | `0.5` | Minimum classifier score to consider an item (high-recall path). |

**`speech`** — local Whisper transcription (see [Speech-to-Text](#speech-to-text) for detail)

| Key | Default | What it does |
|-----|---------|--------------|
| `model` | `small.en` | `base.en` / `small.en` / `medium.en` / `large-v3`. Bigger = more accurate, slower. |
| `compute_type` | `int8` | `int8` / `int8_float16` / `float16` / `float32`. |
| `device` | `auto` | `auto` / `cpu` / `cuda` (GPU via CUDA). |
| `capture_microphone` | `false` | Also record your mic, tagging lines `(me)` vs `(remote)` for attribution. |
| `speaker_naming` | `true` | Post-call, name transcript lines from the roster + intros. |
| `chunk_seconds` | `5` | Audio chunk length. Longer = more context, fewer cuts, more latency. |
| `vad_min_silence_ms` | `500` | Silence gap before splitting a chunk. |
| `no_speech_threshold` | `0.6` | Drop segments Whisper thinks are non-speech. |
| `log_prob_threshold` | `-1.0` | Drop low-confidence decodes. |
| `compression_ratio_threshold` | `2.4` | Drop repetitive / hallucinated decodes. |
| `echo_suppression` | `true` | De-dup speaker bleed across mic + loopback (only with `capture_microphone`). |

> `backend` is no longer needed — local Whisper is the only engine.

**`triggers`** — topic routing

| Key | Default | What it does |
|-----|---------|--------------|
| `client_topics` | – | List of `{pattern: <regex>, action: research\|sizing}` — nudges matching topics into a pipeline. |
| `consultant_hedges` | 6 defaults | Phrases ("let me confirm…") that flag a follow-up to research. |

**`grounding`** — where research looks

| Key | Default | What it does |
|-----|---------|--------------|
| `repo_paths` | `.github/instructions/` | Workspace folders searched for grounding context. |
| `microsoft_learn` | `true` | Query the Microsoft Learn API. |
| `extra_trusted_domains` | – | `{host: weight}` to add / re-weight verified web sources without editing code. |

**`output` / `notifications`**

| Key | Default | What it does |
|-----|---------|--------------|
| `output.auto_save` | `true` | Save summary + deliverables on stop. |
| `output.include_session_summary` | `true` | Include the session summary in the pack. |
| `notifications.sound` | `chime` | `silent` / `chime` / `asterisk` / `exclamation` / `beep`. |

**Global (not per-profile).** LLM model chains are defined in code. Override a
tier without editing anything via an env var in `~/.sidekick/.env`, e.g.
`SIDEKICK_MODEL_DEEP="copilot:claude-opus-4.8,copilot:gpt-4.1"`.

---

## Verify (optional)

Paste these into PowerShell to confirm everything is working:

```powershell
sidekick --help                                                    # CLI works
Get-Content "$env:APPDATA\Code\User\mcp.json" | Select-String sidekick  # MCP registered
code --list-extensions | Select-String sidekick                    # Extension installed
gh auth status                                                     # GitHub token OK
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `sidekick` not found | Close and reopen terminal to refresh PATH |
| MCP server not showing in VS Code | Restart VS Code; check `%APPDATA%/Code/User/mcp.json` has a `sidekick` entry |
| No audio captured | Check system audio is playing through default output device — Sidekick uses loopback capture |
| `gh auth token` fails | Run `gh auth login` and select HTTPS + browser auth |
| ARM64 + faster-whisper | Fully supported via x64 Python emulation (installer handles this automatically) |
| Extension not installed | Run manually: `code --install-extension <path-to-vsix>` — path shown in `sidekick init` output |

---

## Speech-to-Text

Sidekick uses **faster-whisper** running locally on CPU. There are no API keys, no cloud STT, and no audio leaves the device — important for regulated customer engagements.

The default model is `small.en` (~470MB, ~5-7% WER). Override per-customer in `customers.yaml`:

```yaml
myproject:
  speech:
    model: medium.en        # base.en | small.en | medium.en | large-v3
    compute_type: int8      # int8 | int8_float16 | float16 | float32
```

Or via environment variables in `~/.sidekick/.env`:

```env
SIDEKICK_WHISPER_MODEL=small.en
SIDEKICK_WHISPER_COMPUTE=int8
```

> **Note:** Azure Speech support was removed in v0.3.0. See `CHANGELOG.md` for the rationale (diarization was unreliable with Entra ID auth, and `small.en` matches cloud STT accuracy on technical English).

---

## Web Search Grounding

The `research` tool always queries the **Microsoft Learn API** — free, key-less, and authoritative for Microsoft, Fabric, and Azure content. This covers the majority of engagements with **no configuration required**.

To broaden verified results to **non-Microsoft sources** (AWS, Databricks, Delta, Spark, PostgreSQL), set **one** web-search API key in `~/.sidekick/.env`:

```env
# Pick ONE (Tavily is preferred when both are present):
TAVILY_API_KEY=tvly-xxxxxxxx     # https://tavily.com
BRAVE_API_KEY=BSA-xxxxxxxx       # https://brave.com/search/api/
```

| Behaviour | No key | Key set |
|-----------|--------|---------|
| Microsoft / Fabric / Azure verified URLs | Yes | Yes |
| AWS / Databricks / Spark / PostgreSQL verified URLs | No (LLM knowledge only, no citations) | Yes |
| Per-domain routing (e.g. AWS question promotes AWS docs) | n/a | Yes |

**Notes for regulated engagements:**
- The key is **per-machine** and local to `~/.sidekick/.env` — never committed, never in `customers.yaml`. Each consultant sets their own.
- These are **external SaaS**. Only the **query text** leaves the device (no transcript or customer data), and results are filtered to the verified-source allowlist before anything is surfaced.
- Both providers offer free tiers. Leave both keys **unset** unless external-source breadth is explicitly approved for the engagement.
- Customers can add or re-weight a verified source without code changes via `grounding.extra_trusted_domains` in their profile (`{host: weight}`).

---

## Uninstall

### Automated (recommended)

```powershell
sidekick uninstall
```

This removes:
- `~/.sidekick/` — config, cache, session outputs, live alerts
- MCP server entry from `%APPDATA%/Code/User/mcp.json`
- sidekick-notify VS Code extension
- sidekick agent definition from `%APPDATA%/Code/User/prompts/`
- sidekick-copilot uv tool environment

Add `-y` to skip the confirmation prompt:

```powershell
sidekick uninstall -y
```

### Manual (if CLI is already gone)

```powershell
# 1. Remove user data
Remove-Item "$env:USERPROFILE\.sidekick" -Recurse -Force

# 2. Remove uv tool environment
uv tool uninstall sidekick-copilot

# 3. Remove MCP entry — edit %APPDATA%/Code/User/mcp.json
#    Delete the "sidekick" key from "servers"

# 4. Remove VS Code extension
code --uninstall-extension koladimeji.sidekick-notify

# 5. Remove agent definition
Remove-Item "$env:APPDATA\Code\User\prompts\sidekick.agent.md" -Force
```

### Optional: remove shared tools

These are shared by other projects — only remove if you no longer need them:

```powershell
# Remove uv (Python package manager)
irm https://astral.sh/uv/uninstall.ps1 | iex

# Remove GitHub CLI
winget uninstall GitHub.cli
```

### Artifact reference

| Artifact | Location |
|----------|----------|
| Config & data | `~/.sidekick/` |
| Customer profiles | `~/.sidekick/customers.yaml` |
| Session outputs | `~/.sidekick/outputs/<customer>/` |
| Live alerts | `~/.sidekick/live/alerts.jsonl` |
| Cache | `~/.sidekick/cache/` |
| MCP registration | `%APPDATA%/Code/User/mcp.json` |
| VS Code extension | sidekick-notify |
| uv tool env | `uv tool dir` (run to see path) |
