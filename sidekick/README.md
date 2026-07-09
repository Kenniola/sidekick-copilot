# Sidekick — Real-time Meeting Co-pilot

MCP server for GitHub Copilot that listens to meetings, researches questions it hears, suggests what to ask, and generates code prototypes — autonomously while you're on the call.

## Quick Start

### Option A: One-liner (recommended)

```powershell
irm https://raw.githubusercontent.com/Kenniola/sidekick-copilot/main/sidekick/install.ps1 | iex
```

Installs `uv` (if missing) → installs sidekick in an isolated environment → scaffolds `~/.sidekick/` → registers MCP server → installs sidekick-notify extension.

### Option B: uvx (zero-install)

If you have [`uv`](https://docs.astral.sh/uv/) installed, add this to your VS Code User or workspace `mcp.json` — no pip install needed:

```jsonc
// .vscode/mcp.json
{
  "servers": {
    "sidekick": {
      "command": "uvx",
      "args": ["--from", "sidekick-copilot[live]", "sidekick", "serve"]
    }
  }
}
```

Then run `sidekick init` once to scaffold config and install the notification extension.

### Option C: From source (development)

```powershell
cd repo/sidekick
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
sidekick init
```

Install extras: `[live]` (Whisper + audio capture — required for the listen loop), `[dev]` (test + lint), `[all]` (everything).

### LLM Auth

Sidekick uses the **Copilot API** (primary) with **GitHub Models** fallback. Both authenticate via `gh auth token` — no additional env vars needed.

```powershell
gh auth login   # that's it — sidekick auto-detects the token
```

The token refreshes every 30 minutes during long sessions.

#### Model Tiers

| Tier | Model (Copilot API) | Used By | Fallback (GitHub Models) |
|------|---------------------|---------|--------------------------|
| `fast` | gpt-4o-mini | Classifier | gpt-4.1-mini |
| `standard` | claude-sonnet-4.5 | Research, Suggest | gpt-4.1-mini |
| `deep` | claude-opus-4.7 | Prototype, Complex research | DeepSeek-R1 |

All tiers retry with exponential backoff (1s → 2s → 4s) and fall through the chain on 429/5xx.

### Start

```
@sidekick listen                 # default config
@sidekick listen --config acme   # customer profile
```

---

## How It Works

```
@sidekick listen --config acme
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AUDIO CAPTURE                                                     │
│  WASAPI loopback (Teams/Zoom/Meet) → 5s chunks, 16kHz mono        │
│                         │                                          │
│                         ▼                                          │
│         Whisper (faster-whisper, local CPU, on-device)            │
│         Default model: small.en (~470MB, ~5-7% WER)                │
│         Session-relative timestamps; no network, no API keys      │
└───────────────────────────────────────────────────────────────────┘
                │                                                     │
                ▼                                                     │
┌─────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION  (every 10s batch)                                  │
│  gpt-4o-mini classifies transcript → ActionItems + thread updates  │
│  Tracks: conversation threads, meeting phase, key facts, open Qs   │
└───────────────┬─────────────────────────────────────────────────────┘
                │ items with priority score ≥ 0.5
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3-LANE PRIORITY QUEUE                                              │
│  ┌─────────────────┬──────────────────┬──────────────────┐         │
│  │ Fast (×3)        │ Standard (×2)     │ Deep (×1)       │         │
│  │ simple lookups  │ multi-source     │ complex reasoning│         │
│  └────────┬────────┴────────┬─────────┴────────┬─────────┘         │
│           ▼                 ▼                  ▼                    │
│    Research Pipeline   Research Pipeline   Prototype Pipeline      │
│    (MS Learn, repo,    + verified web      Consultant Advisor      │
│     .github/instr.)      sources           (6-step reasoning)      │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                             │
│  Session log → alerts.jsonl → sidekick-notify extension             │
│                                 │                                   │
│                    ┌────────────┴────────────┐                      │
│                    ▼                         ▼                      │
│            Toast (high priority)     Status bar badge               │
│            Toast (standard)          click → @sidekick status       │
│                                                                     │
│  On stop → markdown summary saved to ~/.sidekick/outputs/           │
└─────────────────────────────────────────────────────────────────────┘

  LLM: GitHub Copilot API (zero cost) → GitHub Models (fallback)
  Models: gpt-4o-mini │ Claude Sonnet 4.5 │ Claude Opus 4.7
```

1. **Captures** system audio via WASAPI loopback (Teams, Zoom, Meet)
2. **Transcribes** locally with faster-whisper (default `small.en`, ~470MB) — on-device, no cloud STT
3. **Classifies** questions, hedges ("let me confirm…"), action items
4. **Routes** to 3-lane priority queue (fast / standard / deep)
5. **Executes** research and prototype generation automatically
6. **Auto-stops** after 60s of silence

The `research` tool is for manual ad-hoc questions — the loop handles everything heard on the call.

---

## Tools

| Tool | Shortcut | Purpose |
|------|----------|---------|
| `listen` | — | Start audio capture + autonomous loop |
| `suggest_questions` | `q` / `?` | Ranked questions with claim analysis, corrections, observations |
| `add_context` | — | Inject live context — notes, files, or images (vision LLM) |
| `research` | `r <topic>` | Ad-hoc question (Microsoft Learn + verified web sources, domain-routed; workspace docs) |
| `prototype` | `p <desc>` | Generate code (PySpark, T-SQL, DAX, pipeline) |
| `status` | `s` / `.` | New threads and research since last check |
| `stop` | `x` | End session, save summary |

### `suggest_questions` — Consultant Advisor

6-step chain-of-thought: claim analysis → contradiction detection → gap analysis → risk detection → strategic positioning → timing assessment.

Returns ranked questions with category, impact, rationale, corrections, and observations. Phase-aware (opening → core → deep-dive → wrap-up). Grounded in `.github/instructions/` and past engagement artifacts.

### `add_context` — Live Context Injection

Feed Sidekick information it can't hear: architecture diagrams shown on screen, decisions made in chat, links shared off-mic.

```text
add_context content="Customer is migrating Oracle → Azure SQL, 6-month timeline."
add_context file_path="C:/work/customer-architecture.md"
add_context image_path="C:/work/screenshot.png"
```

| Parameter | Type | Notes |
|-----------|------|-------|
| `content` | text | Free-text notes pasted into the session. |
| `file_path` | path | `.md` `.txt` `.json` `.yaml` `.yml` `.csv` `.sql` — capped at 4000 chars (truncated with marker). |
| `image_path` | path | `.png` `.jpg` `.jpeg` `.gif` `.webp` — capped at 10 MB; processed via vision LLM (gpt-4o) which extracts components, data flows, technologies, integration points, and visible text. |

At least one input is required; all three can be combined in one call. Injected documents are stored on the active session and surface in:

- **Classifier prompts** — last 3 documents, 200 chars each (helps the analyst recognise topics it never heard spoken aloud).
- **Grounding context** — last 5 documents, 1500 chars each (used by `suggest_questions`, `research`, and `prototype`).

Response echoes a one-line summary plus the running `Total context documents` count.

---

## Customer Profiles

Single-file system at `~/.sidekick/customers.yaml`. Profiles deep-merge over `default.yaml` — only specify overrides.

```yaml
acme:
  customer: Acme Corp
  participants:
    consultant: ["Your Name"]
  domains: [Microsoft Fabric, Data Warehousing]
  sensitivity:
    trigger_threshold: 0.6
  triggers:
    client_topics:
      - pattern: "migration|cutover|timeline"
        action: research
```

Config search order: `$SIDEKICK_CONFIG_DIR/` → `~/.sidekick/customers.yaml` → `~/.sidekick/configs/` → package `default.yaml`.

Lists replace wholesale (not merged). Run `sidekick init` to scaffold with a commented template.

---

## Notifications

### Built-in
Findings trigger the Windows default notification chime (`winsound.MessageBeep`, respects **Notification volume** in Sound Settings) and append to `~/.sidekick/live/alerts.jsonl`. Every tool response auto-surfaces new findings via a preamble banner.

Sound is config-driven via `notifications.sound` in `default.yaml` or any customer profile:

```yaml
notifications:
  sound: chime     # silent | chime | asterisk | exclamation | beep
```

| Value | Behaviour |
|-------|-----------|
| `silent` | No sound. |
| `chime` (default) | Standard Windows notification (`MB_OK`). Subtle. |
| `asterisk` | Windows "Information" chime (`MB_ICONASTERISK`). |
| `exclamation` | Windows "Attention" chime (`MB_ICONEXCLAMATION`). |
| `beep` | Legacy raw 800 Hz / 200 ms tone (`winsound.Beep`) — louder, plays at system master volume. |

All values silently no-op on macOS/Linux.

### VS Code Extension (sidekick-notify)
Optional companion extension in `repo/sidekick-notify/` — polls `alerts.jsonl` and shows VS Code toast notifications (the one-line **answer card** plus an **Open Source** button when a finding cites a URL) with a status bar badge.

```powershell
code --install-extension repo/sidekick-notify/sidekick-notify-0.2.0.vsix
```

Click the status bar badge to open `@sidekick status` in chat.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GITHUB_TOKEN` | `gh auth token` | Override token (e.g. CI). Normally auto-detected. |
| `SIDEKICK_WORKSPACE_ROOT` | CWD | Workspace root for repo search and grounding |
| `SIDEKICK_HOME` | `~/.sidekick/` | Override user directory |
| `SIDEKICK_CLASSIFY_INTERVAL` | `10` | Seconds between classifier calls |
| `SIDEKICK_WHISPER_MODEL` | `small.en` | Whisper model size (`base.en` · `small.en` · `medium.en` · `large-v3`) |
| `SIDEKICK_WHISPER_COMPUTE` | `int8` | CTranslate2 compute type (`int8` · `int8_float16` · `float16` · `float32`) |
| `TAVILY_API_KEY` | _(unset)_ | Optional. Enables live web search via [Tavily](https://tavily.com). When unset, research still runs against Microsoft Learn. |
| `BRAVE_API_KEY` | _(unset)_ | Optional. Fallback web-search provider ([Brave Search API](https://brave.com/search/api/)), used only if `TAVILY_API_KEY` is not set. |

Speech config (model, compute_type) can also be set per-customer in `customers.yaml`.

---

## CLI

```
sidekick init          # Scaffold ~/.sidekick/ and register MCP server
sidekick serve         # Run MCP server (called by mcp.json)
sidekick list-configs  # Show available profiles
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  VS Code Copilot Chat ←→ @sidekick agent ←→ MCP Server  │
├──────────────────────────────────────────────────────────┤
│  Audio Capture → Speech-to-Text → Classifier → Queue    │
│  (WASAPI)        (Whisper, local) (LLM)       (3-lane)  │
│                                                          │
│                              ├── Research Pipeline       │
│                              ├── Prototype Pipeline      │
│                              └── Consultant Advisor      │
│                                   + grounding context    │
│  Meeting Context ←→ Session Log → ~/.sidekick/outputs/   │
│  Notifications → alerts.jsonl → sidekick-notify ext      │
└──────────────────────────────────────────────────────────┘
```

Research searches: workspace docs (keyword + content scoring) → `.github/instructions/` (content-aware) → live web (Microsoft Learn API always, plus an optional Tavily/Brave provider) ranked by a verified-source trust map with per-domain routing.

### v0.2.0 Optimisations

- **Domain auto-detection** — LLM analyses first 30 transcript lines at batch 3 to detect domains; merges with config and invalidates grounding cache
- **Grounding context** — `suggest_questions` loads team standards and past engagement artifacts via a shared httpx client (no per-call client creation)
- **`add_context` tool** — inject notes, files (.md/.txt/.py/.json, 4KB cap), or images (base64 → vision LLM extraction, 10MB cap) mid-session
- **Smart dedup** — priority queue checks new questions against last 10 completed outputs via fast-tier LLM; duplicates are re-researched with enriched context (previous answer appended) rather than skipped
- **Per-domain source routing** — live web results (Microsoft Learn + optional Tavily/Brave) are filtered to a verified-source trust map and ranked so Microsoft docs stay high while the question's detected domain lifts its preferred sources (e.g. an AWS question promotes `docs.aws.amazon.com`). Only verified URLs are surfaced for citation. Replaces the retired Bing Web Search API.
- **Dynamic prompts** — hardcoded "Microsoft Fabric" replaced with detected domains across classifier, research synthesis, and advisor prompts
- **Thread detection** — explicit rules in analyst prompt for topic-shift detection; classifier prompt enriched with injected context documents
- **Grounding cache** — 5-minute TTL cache on `_build_grounding_context()` via `asyncio.to_thread()`

### Resilience

- Per-chunk error handling — bad LLM responses don't crash the session
- Consecutive error threshold (5) before the loop stops
- JSON hardening — markdown fence stripping, unknown field filtering
- Silence detection and hallucination guards (no_speech_prob, repetition, VAD)
- **Auto-stop** — 60 seconds of silence triggers automatic shutdown
- **Transcript batching** — 10s classification window reduces LLM calls by ~50%

---

## Project Structure

```
repo/sidekick/
├── pyproject.toml              # Package config (sidekick-copilot)
├── install.ps1                 # Windows bootstrap installer
├── configs/                    # Bundled as package data
│   ├── default.yaml            # Factory defaults
│   └── _template.yaml          # Starter template for customers.yaml
└── src/sidekick/
    ├── server.py               # MCP server — 8 tools + background loop + capture + notifications
    ├── llm.py                  # Multi-backend LLM client + vision API (GitHub/Azure/Anthropic)
    ├── config.py               # Config loader — user-local + package defaults
    ├── cli.py                  # CLI entry points (init, serve, list-configs)
    ├── transcript/
    │   ├── audio_capture.py    # WASAPI loopback audio capture
    │   └── speech_recogniser.py# Local Whisper (faster-whisper) backend
    ├── analyst/
    │   ├── classifier.py       # LLM-powered transcript analyser
    │   ├── context.py          # Meeting state + TranscriptLine
    │   └── prompts.py          # Analyst + consultant advisor (7-step chain-of-thought)
    ├── queue/
    │   └── priority_queue.py   # 3-lane async priority queue
    ├── actions/
    │   ├── research.py         # Multi-source research pipeline with repo search
    │   └── prototype.py        # Code generation pipeline
    └── output/
        └── session_log.py      # Session log + summary generation

~/.sidekick/                    # User-local directory (created by sidekick init)
├── customers.yaml              # Customer profiles
├── outputs/                    # Session logs per customer
└── configs/                    # Individual config file fallback
```

---

## User Directory (`~/.sidekick/`)

All user-local state lives in `~/.sidekick/`:

| Path | Purpose |
|------|---------|
| `customers.yaml` | Customer profiles (single-file, multi-profile) |
| `outputs/<customer>/` | Session logs per customer |
| `live/alerts.jsonl` | Proactive notification log (appended by background loop) |
| `configs/` | Individual config files (fallback if not using customers.yaml) |

Override the location with `SIDEKICK_HOME`.
