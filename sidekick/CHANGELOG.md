# Changelog

All notable changes to sidekick-copilot are documented in this file.

---

## [Unreleased]

### Fixed
- **`sidekick uninstall`/`init` crashed on non-UTF-8 consoles.** The CLI prints
  status glyphs (`✓`, `⚠`, `—`); on a cp1252 console (common on Windows,
  especially when output is piped) that raised `UnicodeEncodeError` and aborted
  the command mid-run. The CLI now forces UTF-8 on stdout/stderr at startup.
  (`tests/test_cli_install.py`)
- **Stale bundled extension on clean installs.** `sidekick init` installs the
  extension force-included in the package, which had drifted to 0.3.0 while the
  source was 0.7.1 — so fresh machines got an old feed UI. Rebundled 0.7.1.
- **Cross-pass duplicate findings.** The adjudicator now carries **session
  memory** of questions already surfaced and suppresses re-asks even when
  reworded/rescoped (e.g. the six "is there a Fabric-specific advantage to
  GitHub over ADO" variants collapse to one). LLM semantic judgement plus a
  lexical backstop; applies on the fallback path too. (`tests/test_adjudicator.py`)
- **"Research in Chat" did nothing.** It opened the chat with an `@sidekick`
  participant that doesn't exist (Sidekick is an MCP server, not a chat
  participant), so the query failed. Now sends a clean, single-line research
  instruction the agent acts on with the Sidekick research tool. (`sidekick-notify`)
- **Auto-suggest (`[ask]`) not appearing.** Feature was correct but off by
  default; enable it per profile with `sensitivity.auto_suggest: true`.

### Added
- **Phase 9.3 (accuracy spec) — proactive advisor (opt-in).** When
  `sensitivity.auto_suggest` is on, a slow-cadence background pass occasionally
  surfaces **one** high-impact question to ask the client, shown in the feed as
  an `[ask]` card (💡). Off by default so it never adds in-call noise; capped at
  one suggestion per cadence, deduped against recent suggestions, and never
  competes with research. (`tests/test_engine.py`)
- **Phase 9.2 (accuracy spec) — feed prettify (extension).** Feed rows are now
  **numbered**, have **markdown stripped** from their plain-text labels (no more
  literal `**`/`##`), carry a **confidence tag** (HIGH/MED/LOW) and a
  **confidence-coloured icon** (green/amber/red; unseen high-priority findings
  keep the orange attention colour). Detail previews are cleaned too, with the
  full markdown still available on hover. (`sidekick-notify`)
- **Phase 9.1 (accuracy spec) — keyless web search.** DuckDuckGo (`ddgs`) is now
  the default web-search provider, so **non-Microsoft results work with no API
  key and no user setup** (a Tavily/Brave key, per-user or a shared org key,
  still takes precedence when present). Results pass through the existing
  verified-source trust map + relevance ranking, so only reputable hosts
  surface. Best-effort — degrades to Microsoft Learn + model knowledge on
  rate-limit/failure. (`tests/test_research_routing.py`)
- **Phase 8 (accuracy spec) — research quality + feed clarity.** Acts on the
  second customer test.
  - **Relevance-aware source ranking (8.1).** Web hits are now scored by topical
    relevance *and* source trust, so an off-topic high-trust page (e.g. a stray
    `learn.microsoft.com` Cosmos DB result for a Terraform question) is demoted
    below an on-topic one, and a relevant reputable non-Microsoft source can
    outrank an off-topic Microsoft page. (`tests/test_research_routing.py`)
  - **Extended verified sources (8.2).** Added HashiCorp/Terraform and other
    reputable technical hosts to the trust map + routing (Microsoft still
    outranks by trust). *Non-Microsoft results require a web-search key
    (`TAVILY_API_KEY`/`BRAVE_API_KEY`) — Microsoft Learn alone only returns
    Microsoft content.*
  - **Relevance-floored citations (8.3).** Only sources clearing a relevance
    floor are cited, so a weakly-matched URL is dropped rather than surfaced as
    a wrong citation.
  - **Real confidence (8.4).** The research result's confidence is now parsed
    from the model's stated HIGH/MEDIUM/LOW instead of a hardcoded "medium".
  - **Feed clarity (8.5/8.6).** Feed rows now lead with the **question**; the
    drill-down shows the **full answer** (untruncated, in the hover), rationale,
    source, and real confidence; and **"Research in Chat"** opens the finding as
    `@sidekick research <question>` instead of the generic status.
    (`sidekick-notify` v0.6.0)
- **Phase 7 (accuracy spec) — LLM speaker-naming.** A best-effort, post-call
  pass attributes transcript lines to named participants using conversational
  cues (self-introductions, direct address, turn-taking) from the roster
  (`config.consultant_names` / `client_names` + detected participants). Runs at
  `stop`, off the live path, so the transcript, summary, and deliverables read
  with names instead of a single `(audio)` tag. Conservative and partial — only
  confident lines are labelled; any failure leaves the source tags intact.
  Toggle with `speech.speaker_naming` (default on). New `analyst/speakers.py`
  (`build_roster`, `name_lines`) + `engine.name_speakers`.
  (`tests/test_speakers.py`, `tests/test_config_merge.py`)
- **Phase 6 (accuracy spec) — relevance & accuracy engine.** Acts on the customer
  08-Jul session study; all behaviour improvements, no new config.
  - **Classifier precision (6.1).** The analyst prompt now explicitly excludes
    the consultant's own statements/coaching, statements of intent, and garbled
    fragments — surface only genuine client questions or verifiable claims.
    (`tests/test_analyst_prompt.py`)
  - **Action-item capture (6.2).** `action_item` classifications now populate
    `context.action_items` (deduped), so the deliverables table is filled
    instead of showing "No action items". (`tests/test_deliverables.py`)
  - **Follow-up hygiene (6.3).** The deliverables follow-up list drops garbled
    fragments and near-duplicates and is capped, so the customer-facing batch
    reads cleanly. (`tests/test_deliverables.py`)
  - **Enrichment restraint (6.4).** A duplicate question answered within a 90 s
    cooldown is no longer re-researched — cutting the repeated `[ENRICHED]`
    churn seen in the customer session. (`tests/test_priority_queue.py`)
- **Phase 5 (accuracy spec) — feed UX & session hygiene.** Acts on the customer
  08-Jul test feedback (`sidekick-notify` v0.5.0).
  - **Category tags + drill-down (5.1/5.2).** Feed rows now show a `[type]` tag
    beside the icon and are **expandable** — click a finding to reveal its
    rationale, source (clickable), file, priority/confidence, and a "View in
    Chat" action, instead of the old generic status jump.
  - **Session hygiene (5.3).** `listen` archives the prior `alerts.jsonl` to
    `live/archive/` (`notifier.rotate_alerts`), so each meeting starts with a
    clean feed and the file can't grow unbounded; the extension clears the feed
    on the new session. (`tests/test_notifier.py`)
  - **Stable enrichment id (5.4).** `notifier._stable_id` strips the
    `[ENRICHED] … (previous answer: …)` wrapper so re-research supersedes the
    original feed row instead of stacking duplicates. (`tests/test_notifier.py`)
  - Pure feed logic (`typeTag`, `detailNodes`, `clear`) covered by
    `node --test` (15 cases).
- **Phase 4 (accuracy spec) — deep-default answers + self-critique.** Trades
  latency for answer accuracy; both opt-in.
  - **Deep-answer routing (A3).** `sensitivity.answer_tier: "deep"` (and, by
    default, `accuracy_mode`) routes every substantive answer to the deep lane
    and the deep model, instead of the complexity-based fast/standard tiers.
    (`tests/test_priority_queue.py`)
  - **Self-critique (A3, opt-in).** `sensitivity.self_critique: true` runs the
    research synthesis as draft → critique-against-sources → refined final
    answer; degrades to the draft on any failure so an answer is never lost.
    (`tests/test_research_routing.py`, `tests/test_config_merge.py`)
- **Phase 3 (accuracy spec) — notification focus.** Replaces the toast-per-finding
  overload with a calm, persistent feed and urgent-only toasts.
  - **Sidekick Feed (B2)** — a persistent activity-bar TreeView (`sidekick-notify`
    v0.4.0) tails `alerts.jsonl` into a live, scrollable list with per-type
    icons, relative timestamps, rationale tooltips, and click-to-open
    (source / file / chat). The feed is the durable record, so toasts no longer
    need to persist.
  - **Priority-gated toasts (B1)** — only `critical`/`high` findings raise a
    toast now; everything else lives silently in the feed. The status-bar badge
    counts **unseen high-priority** items only.
  - **Supersede / dedup / TTL (B3)** — a new alert with an existing `id` updates
    its feed row in place (no re-toast); a newer answer on the same `thread_id`
    dims older ones; findings older than 10 min render as "stale".
  - **Alert schema (B4)** — `notifier` now emits a stable `id`, the adjudicator
    `rationale`, and `thread_id` on every alert. (`tests/test_notifier.py`)
  - **Pure, tested feed logic (B5)** — `feedModel.ts` holds all the
    add/supersede/dedup/badge/stale logic with no `vscode` import, covered by
    11 `node --test` unit tests; `npm test` runs them.
- **Phase 2 (accuracy spec) — structural transcript quality + speaker
  correctness.** Improves the raw transcript the classifier reads, and makes
  dual-capture speaker attribution robust. Defaults preserve current behaviour.
  - **Decode-threshold tuning (C2).** `vad_parameters` (min-silence),
    `no_speech_threshold`, `log_prob_threshold`, and
    `compression_ratio_threshold` are now passed to faster-whisper to cut
    hallucinations and edge clipping — all configurable under `speech:`.
    (`tests/test_speech_recogniser.py`)
  - **Configurable chunk length (C2).** `speech.chunk_seconds` (default 5) —
    longer chunks give Whisper more context and fewer mid-utterance boundary
    cuts, wired into the audio capture. (`tests/test_config_merge.py`)
  - **Cross-speaker echo suppression (C3 Tier 1).** When mic + loopback capture
    both pick up the same utterance (speaker bleed), the duplicate under the
    other speaker tag is dropped — within a 2 s window, similarity ≥ 0.85, and
    only for utterances ≥ 12 chars so genuine short confirmations ("Yes.")
    always survive. Off automatically for single-source capture; toggle via
    `speech.echo_suppression`. (`tests/test_speech_recogniser.py`)

- **Phase 1 (accuracy spec) — two-stage relevance pipeline.** A high-recall fast
  detector now feeds a periodic **deep-tier relevance adjudicator** that surfaces
  only the few questions genuinely worth the consultant's attention, each with a
  one-line rationale tied to the engagement's objectives. Off by default
  (`sensitivity.accuracy_mode`), so existing behaviour is unchanged.
  - **`analyst/adjudicator.py`** — `adjudicate()` runs a deep-tier reasoning
    pass over the fast classifier's candidates, merges near-duplicates,
    re-scores, caps at `max_surfaced_per_pass` (3), gates at `surface_threshold`
    (0.7), and degrades to a deterministic threshold filter on any error.
    `ActionItem` gains a `rationale` field. (`tests/test_adjudicator.py`)
  - **Engagement objectives** (`objectives:` / `context.objectives`) — set via an
    `add_context "goal: …"` note (highest priority), seeded from the profile, or
    auto-inferred from the opening minutes (`engine.infer_objectives`). The
    adjudicator scores relevance against them. (`tests/test_objectives.py`)
  - **Engine cadence** — in accuracy mode candidates accumulate and flush through
    the adjudicator every `adjudicator_interval_seconds` (40, a configurable
    *ceiling*), with an early flush on a critical hedge when
    `adjudicator_pause_flush` is on. (`tests/test_engine.py`)
  - New `sensitivity` config: `accuracy_mode`, `adjudicator_interval_seconds`,
    `adjudicator_pause_flush`, `max_surfaced_per_pass`, `surface_threshold`,
    `answer_tier` (wired in Phase 3), plus `objectives:` — all documented in
    `configs/_template.yaml`. (`tests/test_config_merge.py`)

- **Phase 0 (accuracy spec) — STT model benchmark & selection.**
  - **`sidekick benchmark-stt`** measures each candidate Whisper model's
    real-time factor (RTF) on the actual machine — from a `--audio` file or a
    short loopback recording — and recommends the most accurate model that
    stays under the RTF threshold (default 0.7). Pure decision logic
    (`real_time_factor`, `recommend_model`, `run_benchmark`) lives in
    `transcript/benchmark.py` with the model/audio I/O injected, so it is
    unit-tested offline. (`tests/test_benchmark_stt.py`)
  - **`sidekick init --stt-model <model>`** persists the chosen model into
    `~/.sidekick/.env` (idempotent; replaces any commented default). `.env`
    guidance now highlights `distil-large-v3` (≈ large-v3 accuracy, 6.3× faster,
    low hallucination) as the recommended target. (`tests/test_cli_stt.py`)
  - **Per-speaker repetition guard (C2.3).** The Whisper hallucination
    repetition filter (`_last_text`/`_repeat_count`) is now keyed by speaker, so
    one speaker's repeated short utterance (e.g. "Yes.") can no longer suppress a
    different speaker's identical line. (`tests/test_speech_recogniser.py`)

- **Phase 5f — per-call tailoring levers.** Two config surfaces that sharpen accuracy for a specific engagement, both inert by default.
  - **`glossary:`** (list, on the customer profile) — engagement proper nouns (project / team / product names) seeded verbatim into the Whisper vocabulary prior at high weight via the new `Vocabulary.seed_terms()`, so they are recognised from the first chunk (before in-session adaptation has anything to learn from) and outrank derived seed terms. Unlike `seed()`, multi-word phrases are trusted as-is rather than mined out of free text. (`tests/test_vocabulary.py`, `tests/test_config_merge.py`)
  - **`stt_corrections:`** (mapping `"heard" → "meant"`, on the customer profile) — appended to the analyst system prompt by the new `build_analyst_system_prompt(config)` so the LLM un-mangles a customer's specific jargon on top of the built-in general examples. The classifier builds the prompt once per session. (`tests/test_analyst_prompt.py`)
  - Both options are documented in `configs/_template.yaml` alongside the existing `speech.capture_microphone` (5d) speaker-attribution toggle.

- **Config-driven LLM models** — per-tier model fallback chains are now defined in `configs/default.yaml` under a new `models:` block (`fast` / `standard` / `deep`, each an ordered list of `"provider:model"` strings) instead of being hard-coded in `llm.py`. A new `ModelsConfig` dataclass resolves them, and `set_active_models()` registers the active config so every `call_llm(tier=…)` call honours it without threading config through each call site.
  - **Env override:** `SIDEKICK_MODEL_<TIER>` (e.g. `SIDEKICK_MODEL_DEEP="copilot:claude-opus-4.8,copilot:gpt-4.1"`) swaps a tier's chain at runtime with no YAML edit.
  - **`sidekick models [profile]`** CLI command prints the resolved chain per tier (showing primary vs fallback and any active env override).
  - `call_llm()` gains an optional `chain` parameter for explicit overrides (used by tests). Code defaults in `llm._TIER_CONFIG` are preserved as the standalone fallback. `tests/test_models_config.py` covers parsing, defaults, env override, YAML override, and `call_llm` integration.
  - **Note:** both providers (`copilot`, `github_models`) assume an OpenAI-compatible `/chat/completions` shape. Genuinely different APIs (Anthropic-native, Azure OpenAI) would need a per-provider adapter — not yet implemented.

### Changed

- **`stop` no longer overflows the chat with the full deliverables pack.** A real post-call pack (LLM email + tables) runs to ~13&nbsp;KB, which the chat host spills to a tool-result overflow file the agent can't read — so the deliverables never rendered. `stop` now always persists the *full* pack to disk (new `save_deliverables(..., force=True)`, so it saves even when `output.auto_save` is off) and inlines only a bounded **digest**: a clipped email preview plus the short, deterministic action-item and follow-up sections, with a pointer to the saved file. The session summary and the whole `stop` response are hard-bounded (`_MAX_SUMMARY_CHARS` / `_MAX_STOP_RESPONSE_CHARS`) so the response always renders inline. New `DeliverablesPack` (`full_markdown()` / `inline_digest()`) and `build_deliverables()`; `generate_deliverables()` retained as a thin wrapper. (`tests/test_deliverables.py`)

- **`install.ps1`** package source is no longer a `TODO` — it defaults to the private Git repo and honours a `SIDEKICK_REPO_URL` env override. The same URL is centralised in `server.py` as `_REPO_URL` / `_install_hint()` so the install hint and installer stay in sync.

- **Phase 2 — `server.py` consolidation (behaviour-preserving refactor).** The ~1,140-line `server.py` was decomposed into focused, unit-tested modules. Each slice was committed separately with characterization tests written *before* the extraction (test-first), so the structural moves are verifiably regression-free. Test count grew from 48 → 109.
  - **`parse_llm_json` (2a)** — a single tolerant JSON parser (bare object/array, ```` ```json ```` fences, stray `json` tag, surrounding whitespace) now lives in `llm.py` and replaces four divergent ad-hoc parsers across `server.py`, `classifier.py`, and `priority_queue.py`. (`tests/test_parse_llm_json.py`)
  - **`output/notifier.py` (2b)** — the audible-alert + `alerts.jsonl` audit logic moved out of `server._notify` into a testable module (`play_sound` / `write_alert` / `notify`); `server._notify` is now a thin wrapper resolving the configured sound. (`tests/test_notifier.py`)
  - **`grounding.py` (2c)** — the ~130-line grounding-context file I/O moved into a pure `build_grounding_context(config, context)` function. (`tests/test_grounding.py`)
  - **`session_state.py` (2d)** — the ~15 module globals collapsed into a single `SessionState` dataclass whose attributes are mutated in place, removing every `global` statement; `_init_session` uses `SessionState.reset()`. (`tests/test_session_state.py`)
  - **`engine.py` (2e, partial)** — `detect_domains` + `classify_and_dispatch` extracted, taking `SessionState` and a `notify` callable explicitly so they are testable with mocks. The live audio loop (`_run_listen_loop`) stays in `server.py` for now; its extraction is deferred to Phase 3 pending a live-loop test harness. (`tests/test_engine.py`)

- **Phase 3 — test hardening + live-loop extraction.** Completed the Phase 2 carry-over and broadened automated coverage of the highest-risk subsystems. Test count grew 109 → 179.
  - **Live audio loop extracted (completes 2e)** — `_run_listen_loop` moved from `server.py` into `engine.py`, split into `run_listen_loop` → `_initialise_capture` + `_consume_audio` so the batching, silence-timeout, error-budget, and cleanup logic is testable with fake capture/recogniser components (no audio hardware, no clock sleeps). Loop tuning constants (`MAX_CONSECUTIVE_ERRORS`, `SILENCE_TIMEOUT_SECS`, `AUDIO_POLL_SECS`) are now module-level for monkeypatching. `server.listen` launches `engine.run_listen_loop(_state, _notify)`. (`tests/test_listen_loop.py`, 12 tests)
  - **Priority queue** — lane routing, in-queue merge (`batch_with` / `related_to`), deterministic dedup enrichment, per-call concurrency cap, success/timeout/exception handling in `process_ready`, and stale-item expiry. (`tests/test_priority_queue.py`, 15 tests)
  - **LLM tier routing + fallback** — chain resolution (builtin tier, active-models override, explicit chain, unknown-provider skip), retry-then-success on 429/5xx, connection-error retry budget then provider fallback, non-retryable 4xx skip, all-fail `RuntimeError`, and per-provider HTTP request shape via a faked pooled client. (`tests/test_llm_routing.py`, 13 tests)
  - **Config + classifier + session log** — `_deep_merge` semantics and `_parse_config` (flat vs nested participants, legacy `azure`→`whisper`, models/notifications parsing); `AnalystResponse.from_json` fence-stripping, unknown-key filtering, and malformed-item skipping; `SessionLog` record defaults, summary grouping, and `format_outputs` windowing. (`tests/test_config_merge.py` + `tests/test_classifier_parse.py` + `tests/test_session_log.py`, 30 tests)

- **Phase 4 — killer features.** Higher-signal output and lower perceived latency for live calls. Test count grew 179 → 228.
  - **Answer-card toast (4a)** — `notifier.write_alert` now records a one-line `answer` (lead text, Sources block stripped, clipped to ~160 chars on a word boundary) and the first `source` URL alongside each alert. The `sidekick-notify` VS Code extension (bumped to `0.2.0`) shows the answer as the toast headline and offers an **Open Source** button when a URL is present. (`tests/test_notifier.py`)
  - **Post-call deliverables (4b)** — `stop` now generates a `deliverables_<ts>.md` containing a draft follow-up email (LLM, British English, no invented commitments), a deterministic action-item table, and a "couldn't-answer-live" follow-up research batch (open questions/threads not addressed during the call). Email drafting degrades gracefully to a placeholder on LLM failure. Opt out with `stop(deliverables=False)`. (`output/deliverables.py`, `tests/test_deliverables.py`)
  - **Streaming synthesis (4c)** — new `llm.stream_llm()` mirrors `call_llm`'s tier routing / retry / provider fallback but yields content deltas (once any delta is emitted, a mid-stream failure re-raises rather than retrying, to avoid duplication). The research pipeline gains an opt-in `on_lead` callback: on the background path the lead answer is surfaced via the answer-card toast as soon as it streams, rather than after the full synthesis + Sources block. The queue threads `notify` through and flags `early_notified` so the engine skips a duplicate final notification; the synchronous `research` tool path (no `on_lead`) is byte-identical to before. **Honest constraint:** MCP stdio returns a single tool result, so partial tokens cannot stream into the Copilot Chat window — the latency win is on the background research/answer-card path only. (`tests/test_streaming.py`)
  - **Whisper device auto-detect (4d)** — `device: auto | cpu | cuda` (config / `SIDEKICK_WHISPER_DEVICE` env / param). `auto` uses a CUDA GPU when present (compute `float16`) and otherwise CPU (`int8`); explicit `compute_type` is always honoured and a runtime GPU init failure falls back to CPU. **Honest constraint:** the faster-whisper (CTranslate2) backend supports CUDA GPU and CPU only — there is no NPU/DirectML path, so this covers GPU-vs-CPU rather than NPU. VAD gating was already enabled via `vad_filter=True`. (`tests/test_speech_recogniser.py`)

### Fixed

- **Uninstall self-lock on Windows** — `sidekick uninstall` runs from the `sidekick.exe` that lives *inside* the uv tool environment it is trying to delete, so `uv tool uninstall` hit a Windows file lock, failed silently, and printed a misleading "not in uv tools (already removed)" while leaving a corrupted `%APPDATA%\uv\tools\sidekick-copilot` behind. The removal is now delegated to a detached helper that waits for this process to exit before running `uv tool uninstall` (new `_running_inside_uv_tool()` detection + `_uninstall_uv_tool()`); the non-self-locked path reports the real outcome instead of always claiming success. (`tests/test_cli_install.py`)

- **MCP registration now pins `SIDEKICK_WORKSPACE_ROOT`** — `_register_mcp_server` writes `"env": {"SIDEKICK_WORKSPACE_ROOT": "${workspaceFolder}"}` into the `mcp.json` server entry. `build_grounding_context()` and the research pipeline both default this to `"."`; without it they resolved relative to the server's process cwd and silently skipped the team's `.github/instructions` standards. VS Code substitutes the open workspace at launch. (`tests/test_cli_install.py`)

- **Removed stale Azure Speech branches** left over from the v0.3.0 removal: the `listen` banner and `status` tool referenced `_config.speech.azure_region` (no longer a field) behind a now-unreachable `backend == "azure"` guard, and the `listen` docstring still said "Whisper or Azure". Backend label is now simply "Whisper (local)".

### Performance

- **LLM connection pre-warm** — `listen` now kicks off a best-effort `llm.prewarm()` task that acquires the GitHub token and opens a pooled TLS connection to the Copilot host, so the first classifier/research call skips DNS + TCP + TLS setup. All failures are swallowed.

- **Token-budgeted deep-tier prompt (2f)** — `suggest_questions` now caps each prompt block via `prompt_budget.clip` (transcript 6k chars keep-tail, threads 2k, research 2.5k, grounding 4k keep-head) so a long meeting can't blow the deep model's context window or inflate latency. (`tests/test_prompt_budget.py`)
- **Deterministic question dedup (2g)** — `PriorityQueue.enqueue` no longer makes a second fast-tier LLM round-trip per item. The new `dedup` module combines token-set Jaccard + `difflib` sequence ratio (≥0.8) against the last 10 completed questions; `_find_completed_duplicate` is now synchronous and deterministic, with zero added latency or tokens. (`tests/test_dedup.py`)


---

## [0.3.0] — 2026-06-10

### Removed

- **Azure Speech backend** — entirely removed from the codebase, including `AzureSpeechRecogniser`, the `[azure]` install extra, `azure-identity` and `azure-cognitiveservices-speech` dependencies, all `AZURE_SPEECH_*` environment variable handling, `SpeechConfig.azure_*` fields, `speaker_map`, the installer's `azure` feature flag, and the `Azure Speech (Optional)` section from `README.md` and `INSTALL.md`.
  - **Rationale:** real-meeting transcript analysis showed (1) diarization had been silently disabled (the code used `SpeechRecognizer` instead of `ConversationTranscriber` due to `SPXERR_INVALID_ARG` with Entra ID auth), and (2) Azure Speech's only practical advantage over local Whisper was lost. For regulated customer engagements, the on-device privacy posture of Whisper is also decisive.
  - **Migration:** customer YAML profiles with `backend: azure` are auto-rewritten to `whisper` at load time and a warning is logged. Delete the now-unused `AZURE_SPEECH_*` lines from `~/.sidekick/.env`.
- **`offerings` tool and the internal knowledge-base integration** — removed the `offerings` MCP tool, its pipeline module and tests, the proactive offerings background search, the `suggest_questions` offerings fetch, and every offerings reference in the analyst prompt chain, README, and agent definition. Sidekick is now seven tools.
  - **Rationale:** the internal knowledge base is Entra-gated and the sidekick server process cannot authenticate to it. A sidekick (MCP server) cannot call another MCP server directly — both only talk to the host/client — so live offerings would require building and owning a bespoke bridge process that is brittle (hardwired to that schema) and re-introduces the auth complexity removed alongside Azure Speech. The feature delivered no live value and the "auth required" placeholder code was misleading. Topic-relevant delivery guidance is better surfaced by the `research` tool against verified web sources.

### Changed

- **Default Whisper model upgraded from `base.en` to `small.en`** (~150MB → ~470MB, WER ~8-10% → ~5-7%). Real-world transcripts of a 72-minute consulting call showed `base.en` produced visible misrecognitions on technical jargon (Fabric, OneLake, capacity SKUs); `small.en` resolves these.
- **`SpeechRecogniser` Protocol now accepts a `chunk_start_offset: float = 0.0` parameter** so segment timestamps are session-relative (`HH:MM:SS.mmm` reflecting position within the meeting) rather than chunk-relative (always in `[0, 5s]`).
- **`server.py` listen loop now tracks `listen_started_at`** and computes `chunk_start_offset = max(0, time.monotonic() - listen_started_at - chunk_duration)` for every transcription call.
- `SpeechConfig` fields simplified to `backend`, `language`, `model`, `compute_type`.
- `install.ps1` `-Features` parameter restricted to `live` (only supported value).
- `pyproject.toml` `[all]` extra now expands to `[live, dev]`.

### Fixed

- **Chunk-relative transcript timestamps** — segments from every 5-second buffer previously displayed `00:00.000 → 00:05.000` regardless of when in the meeting they occurred, making transcript review of long sessions impossible. All segment timestamps are now meeting-wall-clock-relative.
- `_format_ts()` clamps negative values to `0.0` instead of producing malformed strings.

### Added

- **Per-domain source routing in `research`** — live web results now flow through a single ranker that filters to a verified-source trust map (`_SOURCE_TRUST`) and boosts the question's detected domain's preferred sources (`_DOMAIN_ROUTING`). Microsoft properties keep the highest baseline (engagement verification rule), but an AWS/Databricks/Spark/PostgreSQL question lifts those docs above their baseline so they can rank alongside Microsoft. Non-allowlisted hosts are dropped, so only verified URLs are ever surfaced for citation. New `tests/test_research_routing.py` covers filtering, default Microsoft priority, AWS promotion, dedup, host-anchored matching, and config extension.
- **Live web-search provider (replaces retired Bing)** — `research` now calls [Tavily](https://tavily.com) (`TAVILY_API_KEY`) or, as a fallback, the [Brave Search API](https://brave.com/search/api/) (`BRAVE_API_KEY`), selected by whichever key is present. With no key set, research still runs against the free Microsoft Learn API. Tavily requests are scoped to the verified-source allowlist via `include_domains`; Brave results are filtered post-hoc by the same ranker.
- **`grounding.extra_trusted_domains`** config option (`{host: weight}`) lets a customer profile add or re-weight a verified source without editing code.
- `SIDEKICK_WHISPER_COMPUTE` environment variable (`int8` default; `int8_float16` / `float16` / `float32` supported).
- `tests/test_speech_recogniser.py` regression suite covering `_format_ts`, factory backend fallback, and `chunk_start_offset` propagation.
- **Config-driven notification sound** — new `notifications.sound` setting in `default.yaml` / customer profiles. Accepts `silent`, `chime` (default, standard Windows notification via `MessageBeep(MB_OK)`), `asterisk`, `exclamation`, or `beep` (legacy 800 Hz / 200 ms tone). Replaces the hard-coded 1 kHz / 300 ms `winsound.Beep`, which played at system master volume with no way to soften it. The new default chime respects the **Notification volume** slider in Windows Sound Settings.

---

## [0.2.0] — 2026-06-09

### Added

- **`add_context` tool** — inject text, files (.md/.txt/.json/.yaml/.yml/.csv/.sql, 4000-char cap), or images (.png/.jpg/.jpeg/.gif/.webp, 10MB cap via vision LLM) into the live session. Injected context appears in classifier prompts (last 3 docs, 200 chars each) and grounding context (last 5 docs, 1500 chars each).
- **Domain auto-detection** — fast-tier LLM analyses first 30 transcript lines at classifier batch 3 to detect technology domains. Detected domains merge with config-specified domains and invalidate the grounding cache.
- **Thread detection rules** — explicit `THREAD DETECTION RULES` section in the analyst system prompt guides topic-shift detection, granular thread creation, and thread lifecycle management.
- **Semantic dedup in priority queue** — `_find_completed_duplicate()` compares new questions against last 10 completed outputs via fast-tier LLM. Duplicates are re-researched with enriched context (previous answer appended) rather than skipped.
- **URL filtering for MS Learn** — `_is_useful_url()` rejects shallow URLs (<3 path segments) and training/certification/study-guide pages. Research fetches 8 results, filters, returns top 5.
- **Content-aware instruction search** — `_search_instructions()` now reads first 1500 chars of file content (not just filenames) with weighted scoring (2× filename, 1× content match).
- **Grounding cache** — 5-minute TTL cache on `_build_grounding_context()` via `asyncio.to_thread()`, invalidated on domain detection.
- **`context_documents` field** on `MeetingContext` — accumulates injected context for use by classifier and grounding.
- **`detected_domains` field** on `MeetingContext` — stores auto-detected domains from transcript analysis.

### Changed

- **Shared httpx clients** — `llm.py` now uses global `httpx.AsyncClient` instances per endpoint (`_copilot_client`, `_github_models_client`) with connection pooling, replacing per-call client creation.
- **Parallel I/O in `suggest_questions`** — Eng Hub search and grounding context load concurrently via `asyncio.gather()`. Eng Hub wrapped in `asyncio.wait_for(timeout=10.0)`.
- **Dynamic prompt scoping** — hardcoded "Microsoft Fabric" replaced with `{domain_scope}` template variable in `SYNTHESIS_SYSTEM_PROMPT`, `ANALYST_SYSTEM_PROMPT`, and `CONSULTANT_ADVISOR_PROMPT`. Populated from detected or config-specified domains at runtime.
- **Classifier prompt enrichment** — `_build_prompt()` now includes last 3 `context_documents` (200 chars each) in an `INJECTED CONTEXT` section.
- **Research `_search_ms_learn()`** — fetches 8 results (was 5), applies `_is_useful_url()` filter, returns top 5.
- Tool count: 7 → 8 (added `add_context`).

### Fixed

- Per-call `httpx.AsyncClient` creation causing connection overhead on every LLM call.
- Instruction search matching only filenames, missing relevant content in instruction files.
- Grounding context blocking the event loop (now runs in thread pool).

---

## [0.1.0] — 2026-06-02

### Added

- Initial release with 7 tools: `listen`, `suggest_questions`, `research`, `offerings`, `prototype`, `status`, `stop`.
- WASAPI loopback audio capture with silence detection and auto-stop.
- Whisper (local CPU) and Azure Speech (Entra ID, speaker diarization) backends.
- 3-lane async priority queue (fast/standard/deep) with merge and expiry.
- Multi-source research pipeline: workspace docs, `.github/instructions/`, Microsoft Learn.
- Eng Hub Resource Center VBD/IP offering search.
- 7-step chain-of-thought consultant advisor in `suggest_questions`.
- Proactive notifications via winsound + alerts.jsonl + sidekick-notify extension.
- Customer config profiles with deep merge over defaults.
- Session log saved to `~/.sidekick/outputs/<customer>/` on stop.
- Copilot API primary LLM with GitHub Models fallback, 3-tier routing (fast/standard/deep).
