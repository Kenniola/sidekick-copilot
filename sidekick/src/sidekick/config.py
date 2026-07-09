"""Configuration loader — user-local configs with package-bundled defaults.

Config resolution order:
  1. $SIDEKICK_CONFIG_DIR (if set) — legacy / dev override
  2. ~/.sidekick/customers.yaml — single-file with named profiles
  3. ~/.sidekick/configs/<name>.yaml — individual file fallback
  4. Package-bundled default.yaml — factory defaults

Customer profiles deep-merge over the package default.
"""

from __future__ import annotations

import importlib.resources
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load ~/.sidekick/.env if it exists (GITHUB_TOKEN, model overrides, etc.)
_env_file = Path(os.environ.get("SIDEKICK_HOME", Path.home() / ".sidekick")) / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)


# ---------------------------------------------------------------------------
# User directory
# ---------------------------------------------------------------------------

def get_user_dir() -> Path:
    """Return the Sidekick user directory (~/.sidekick/)."""
    return Path(os.environ.get("SIDEKICK_HOME", Path.home() / ".sidekick"))


def get_cache_dir() -> Path:
    """Return the cache directory (~/.sidekick/cache/)."""
    d = get_user_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_dir(customer: str = "general") -> Path:
    """Return the output directory for a customer (~/.sidekick/outputs/<customer>/)."""
    d = get_user_dir() / "outputs" / customer.lower().replace(" ", "-")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensitivityConfig:
    trigger_threshold: float = 0.5
    noise_filter: str = "medium"
    analyst_interval_seconds: int = 10
    verify_consultant_answers: bool = True
    show_verifications: str = "corrections_only"
    # Two-stage accuracy pipeline (Phase 1 / A1). When ``accuracy_mode`` is on,
    # fast per-chunk classification only produces *candidates*; a periodic
    # deep-tier adjudicator selects the few worth surfacing. All default-off /
    # no-op so existing behaviour is unchanged.
    accuracy_mode: bool = False
    adjudicator_interval_seconds: int = 40  # deep-pass cadence ceiling
    adjudicator_pause_flush: bool = True     # early flush on a critical hedge
    max_surfaced_per_pass: int = 3           # hard cap per adjudicator pass
    surface_threshold: float = 0.7           # precision gate out of the pass
    answer_tier: str = "auto"                # "auto" | "deep" deep-model answers
    self_critique: bool = False              # draft->critique->refine (opt-in)
    # Proactive advisor (Phase 9.3, opt-in). When on, a slow background pass
    # occasionally suggests ONE high-impact question to ask the client, shown
    # in the feed as an ``[ask]`` card. Off by default to avoid in-call noise.
    auto_suggest: bool = False
    auto_suggest_interval_seconds: int = 120  # min gap between suggestions


@dataclass
class QueueConfig:
    fast_lane_max: int = 3
    standard_lane_max: int = 2
    deep_lane_max: int = 1
    stale_expiry_minutes: int = 5


@dataclass
class PhasesConfig:
    opening_minutes: int = 5
    wrapup_keywords: list[str] = field(default_factory=lambda: [
        "next steps", "action items", "to summarise", "wrap up", "before we go",
    ])


@dataclass
class TriggerPattern:
    pattern: str
    action: str
    grounding: str = ""


@dataclass
class TriggersConfig:
    client_topics: list[TriggerPattern] = field(default_factory=list)
    consultant_hedges: list[str] = field(default_factory=lambda: [
        "I'll get back to you",
        "I'll have to check",
        "let me verify",
        "let me confirm",
        "I'm not 100% sure",
        "that's a good question",
    ])


@dataclass
class GroundingConfig:
    repo_paths: list[str] = field(default_factory=lambda: [".github/instructions/"])
    microsoft_learn: bool = True
    # Optional per-customer extensions to the research verified-source trust map,
    # as {host: weight}. Code defaults already cover Microsoft + common partner/OSS
    # docs; use this only to add or re-weight a host without editing code.
    extra_trusted_domains: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputConfig:
    auto_save: bool = True
    include_session_summary: bool = True


@dataclass
class NotificationsConfig:
    """Audible notification settings.

    ``sound`` accepts one of:
      - ``silent``       — no sound at all
      - ``chime``        — standard Windows notification chime (default,
                          ``winsound.MessageBeep(MB_OK)``). Subtle; respects
                          the Notification volume slider in Sound Settings.
      - ``asterisk``     — Windows "Information" chime (``MB_ICONASTERISK``)
      - ``exclamation``  — Windows "Attention" chime (``MB_ICONEXCLAMATION``)
      - ``beep``         — legacy raw 800 Hz / 200 ms square-wave tone
                          (``winsound.Beep``). Plays at system master volume.

    All sounds are no-ops on non-Windows platforms.
    """

    sound: str = "chime"


@dataclass
class SpeechConfig:
    """Local Whisper speech-to-text settings.

    Azure Speech was removed in v0.3.0 — sidekick now uses local Whisper
    exclusively (no network, no API keys). See CHANGELOG for rationale.
    """

    backend: str = "whisper"         # only "whisper" is supported
    language: str = "en-GB"          # informational; Whisper uses "en"
    model: str = "small.en"          # base.en | small.en | medium.en | large-v3
    compute_type: str = "int8"       # int8 | int8_float16 | float16 | float32
    device: str = "auto"             # auto | cpu | cuda (CTranslate2 — no NPU)
    # Dual-source speaker attribution (5d). When False (default) sidekick
    # captures system audio only, tagged "(audio)" — unchanged behaviour. When
    # True it additionally captures the local microphone, tagging remote audio
    # "(remote)" and the local mic "(me)" so the analyst can attribute speech.
    capture_microphone: bool = False
    # Structural transcript quality (Phase 2 / C2). Longer chunks give Whisper
    # more context and fewer mid-utterance boundary cuts; the VAD / decode
    # thresholds are passed through to faster-whisper to cut hallucinations and
    # clipping. Defaults preserve current behaviour.
    chunk_seconds: float = 5.0
    vad_min_silence_ms: int = 500
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    # Cross-speaker echo suppression (Phase 2 / C3 Tier 1). Drops a near-
    # duplicate line from the other capture within a short window (speaker
    # bleed between mic and loopback). No-op unless capture_microphone is on.
    echo_suppression: bool = True
    # Post-call LLM speaker-naming (Phase 7 / C3 Tier 2). Attributes transcript
    # lines to named participants (from the roster + intros) at stop so the
    # transcript/summary/deliverables read with names. Best-effort; degrades to
    # source tags on failure.
    speaker_naming: bool = True


# Canonical default model fallback chains, shared with llm._TIER_CONFIG.
# Each entry is a "provider:model" string; providers are resolved by llm.py
# (currently "copilot" and "github_models").
_DEFAULT_MODEL_CHAINS: dict[str, list[str]] = {
    "fast": [
        "copilot:gpt-4o-mini",
        "github_models:gpt-4.1-mini",
    ],
    "standard": [
        "copilot:claude-sonnet-4.5",
        "copilot:gpt-4.1",
        "github_models:gpt-4.1-mini",
    ],
    "deep": [
        "copilot:claude-opus-4.8",
        "copilot:claude-opus-4.7",
        "copilot:claude-opus-4.6",
        "copilot:gpt-4.1",
        "github_models:DeepSeek-R1",
    ],
}


def _as_bool(value: object) -> bool:
    """Coerce a YAML/env value to bool (accepts true/1/yes/on, case-insensitive)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_model_chain(entries: list[str]) -> list[tuple[str, str]]:
    """Parse ``["provider:model", …]`` into ``[(provider, model), …]``.

    Entries without a ``:`` separator are assumed to target the ``copilot``
    provider. Blank entries are skipped.
    """
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        item = (entry or "").strip()
        if not item:
            continue
        if ":" in item:
            provider, model = item.split(":", 1)
            parsed.append((provider.strip(), model.strip()))
        else:
            parsed.append(("copilot", item))
    return parsed


@dataclass
class ModelsConfig:
    """Per-tier LLM model fallback chains (global, not per-customer).

    Each tier is an ordered list of ``"provider:model"`` strings. The first
    entry is tried first; subsequent entries are fallbacks. Defaults mirror the
    code defaults in ``llm._TIER_CONFIG``.

    An environment variable ``SIDEKICK_MODEL_<TIER>`` (e.g.
    ``SIDEKICK_MODEL_DEEP="copilot:claude-opus-4.8,copilot:gpt-4.1"``) overrides
    the configured list for that tier at resolution time — handy for a quick
    swap without editing YAML.
    """

    fast: list[str] = field(
        default_factory=lambda: list(_DEFAULT_MODEL_CHAINS["fast"])
    )
    standard: list[str] = field(
        default_factory=lambda: list(_DEFAULT_MODEL_CHAINS["standard"])
    )
    deep: list[str] = field(
        default_factory=lambda: list(_DEFAULT_MODEL_CHAINS["deep"])
    )

    def chain(self, tier: str) -> list[tuple[str, str]]:
        """Resolve a tier to a ``[(provider, model), …]`` fallback chain.

        Resolution order: ``SIDEKICK_MODEL_<TIER>`` env var (comma-separated)
        → the configured list for the tier → the ``standard`` list.
        """
        env_override = os.environ.get(f"SIDEKICK_MODEL_{tier.upper()}", "").strip()
        if env_override:
            chain = _parse_model_chain(env_override.split(","))
            if chain:
                return chain

        entries = getattr(self, tier, None)
        if not entries:
            entries = self.standard
        return _parse_model_chain(entries)


@dataclass
class SidekickConfig:
    customer: str = "General"
    description: str = ""
    consultant_names: list[str] = field(default_factory=list)
    client_names: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=lambda: [
        "Microsoft Fabric", "Power BI", "Azure Data Platform",
    ])
    # Per-customer proper nouns / product / project / team names. Seeded into
    # the Whisper vocabulary prior at high weight so they are recognised from
    # the first chunk, before in-session adaptation has anything to learn from.
    glossary: list[str] = field(default_factory=list)
    # Per-engagement speech-to-text corrections, e.g. {"on lake": "OneLake"}.
    # Appended to the analyst system prompt so the LLM un-mangles the customer's
    # specific jargon on top of the built-in general examples.
    stt_corrections: dict[str, str] = field(default_factory=dict)
    # Engagement objectives (Phase 1 / A2) — concrete goals the adjudicator
    # scores relevance against. May be set explicitly here, via an
    # ``add_context "goal: …"`` note at call start, or auto-inferred from the
    # opening minutes when left empty.
    objectives: list[str] = field(default_factory=list)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    triggers: TriggersConfig = field(default_factory=TriggersConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    rules: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence.

    Dicts are merged recursively; all other types (including lists)
    are replaced wholesale by the override value.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_package_default() -> dict:
    """Load the factory default.yaml bundled with the package."""
    try:
        ref = importlib.resources.files("sidekick") / "configs" / "default.yaml"
        return yaml.safe_load(ref.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, TypeError):
        return {}


def _find_customer_profile(name: str) -> dict | None:
    """Search for a customer profile by name.

    Search order:
      1. $SIDEKICK_CONFIG_DIR/<name>.yaml (legacy / dev override)
      2. ~/.sidekick/customers.yaml → profile by name
      3. ~/.sidekick/configs/<name>.yaml (individual file fallback)
    """
    # 1. Legacy config dir override (for development / backward compat)
    config_dir = os.environ.get("SIDEKICK_CONFIG_DIR")
    if config_dir:
        legacy_file = Path(config_dir) / f"{name}.yaml"
        if legacy_file.exists():
            with open(legacy_file, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    user_dir = get_user_dir()

    # 2. Single-file profiles: ~/.sidekick/customers.yaml
    customers_file = user_dir / "customers.yaml"
    if customers_file.exists():
        with open(customers_file, encoding="utf-8") as f:
            all_profiles = yaml.safe_load(f) or {}
        if isinstance(all_profiles, dict) and name in all_profiles:
            profile = all_profiles[name]
            return profile if isinstance(profile, dict) else {}

    # 3. Individual file: ~/.sidekick/configs/<name>.yaml
    individual_file = user_dir / "configs" / f"{name}.yaml"
    if individual_file.exists():
        with open(individual_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return None


def load_config(name: str = "default") -> SidekickConfig:
    """Load a customer config by name, inheriting from package defaults.

    Resolution:
      - Search for a profile matching *name* and deep-merge over defaults.
      - If no profile is found and name is 'default', return package defaults.
      - If no profile is found and name is anything else, raise an error.

    This means you can add a ``default`` profile to customers.yaml to
    customise what ``listen`` uses when no config is specified.
    """
    base_raw = _load_package_default()

    profile_raw = _find_customer_profile(name)
    if profile_raw is not None:
        merged = _deep_merge(base_raw, profile_raw)
        return _parse_config(merged)

    # No profile found — fall back for "default", error for anything else
    if name == "default":
        return _parse_config(base_raw)

    available = list_available_configs()
    raise FileNotFoundError(
        f"Customer profile '{name}' not found.\n"
        f"Available profiles: {', '.join(available) or '(none)'}\n"
        f"Add a profile to: {get_user_dir() / 'customers.yaml'}"
    )


def list_available_configs() -> list[str]:
    """List all available customer profile names."""
    profiles: set[str] = set()

    # From customers.yaml
    customers_file = get_user_dir() / "customers.yaml"
    if customers_file.exists():
        try:
            with open(customers_file, encoding="utf-8") as f:
                all_profiles = yaml.safe_load(f) or {}
            if isinstance(all_profiles, dict):
                profiles.update(all_profiles.keys())
        except Exception:
            pass

    # From individual files
    configs_dir = get_user_dir() / "configs"
    if configs_dir.exists():
        for f in configs_dir.glob("*.yaml"):
            if not f.name.startswith("_"):
                profiles.add(f.stem)

    # From legacy config dir
    config_dir = os.environ.get("SIDEKICK_CONFIG_DIR")
    if config_dir:
        for f in Path(config_dir).glob("*.yaml"):
            if not f.name.startswith(("_", "default")):
                profiles.add(f.stem)

    return sorted(profiles)


def _parse_config(raw: dict) -> SidekickConfig:
    """Parse a raw YAML dict into a SidekickConfig.

    Supports both flat and nested participant keys:
      consultant: "Your Name"           # flat string
      consultant: ["Name1", "Name2"]    # flat list
      participants:                      # nested (legacy)
        consultant: ["Name"]
    Flat keys take precedence over nested participants.
    """
    participants = raw.get("participants", {})
    sensitivity_raw = raw.get("sensitivity", {})
    queue_raw = raw.get("queue", {})
    phases_raw = raw.get("phases", {})
    triggers_raw = raw.get("triggers", {})
    grounding_raw = raw.get("grounding", {})
    output_raw = raw.get("output", {})
    notifications_raw = raw.get("notifications", {})
    speech_raw = raw.get("speech", {})
    models_raw = raw.get("models", {})

    # Resolve consultant names: flat key > nested participants
    consultant = raw.get("consultant") or participants.get("consultant", [])
    if isinstance(consultant, str):
        consultant = [consultant]

    # Resolve client names: flat key > nested participants
    client = raw.get("client") or participants.get("client", [])
    if isinstance(client, str):
        client = [client]

    client_topics = [
        TriggerPattern(**t) for t in triggers_raw.get("client_topics", [])
    ]

    # Normalise legacy backend values. Azure Speech was removed in v0.3.0;
    # the recogniser factory also logs a warning at runtime when a non-whisper
    # value is encountered.
    backend_raw = speech_raw.get("backend", "whisper")
    backend = backend_raw if backend_raw in ("whisper", "", None) else "whisper"

    return SidekickConfig(
        customer=raw.get("customer", "General"),
        description=raw.get("description", ""),
        consultant_names=consultant,
        client_names=client,
        domains=raw.get("domains", ["Microsoft Fabric", "Power BI", "Azure Data Platform"]),
        glossary=[str(t).strip() for t in (raw.get("glossary") or []) if str(t).strip()],
        stt_corrections={
            str(k): str(v)
            for k, v in (raw.get("stt_corrections") or {}).items()
            if str(k).strip() and str(v).strip()
        },
        objectives=[
            str(o).strip() for o in (raw.get("objectives") or []) if str(o).strip()
        ],
        sensitivity=SensitivityConfig(
            trigger_threshold=sensitivity_raw.get("trigger_threshold", 0.5),
            noise_filter=sensitivity_raw.get("noise_filter", "medium"),
            analyst_interval_seconds=sensitivity_raw.get("analyst_interval_seconds", 10),
            verify_consultant_answers=sensitivity_raw.get("verify_consultant_answers", True),
            show_verifications=sensitivity_raw.get("show_verifications", "corrections_only"),
            accuracy_mode=_as_bool(
                os.environ.get(
                    "SIDEKICK_ACCURACY_MODE",
                    sensitivity_raw.get("accuracy_mode", False),
                )
            ),
            adjudicator_interval_seconds=sensitivity_raw.get(
                "adjudicator_interval_seconds", 40
            ),
            adjudicator_pause_flush=_as_bool(
                sensitivity_raw.get("adjudicator_pause_flush", True)
            ),
            max_surfaced_per_pass=sensitivity_raw.get("max_surfaced_per_pass", 3),
            surface_threshold=sensitivity_raw.get("surface_threshold", 0.7),
            answer_tier=str(sensitivity_raw.get("answer_tier", "auto")).lower(),
            self_critique=_as_bool(sensitivity_raw.get("self_critique", False)),
            auto_suggest=_as_bool(sensitivity_raw.get("auto_suggest", False)),
            auto_suggest_interval_seconds=sensitivity_raw.get(
                "auto_suggest_interval_seconds", 120
            ),
        ),
        queue=QueueConfig(
            fast_lane_max=queue_raw.get("fast_lane_max", 3),
            standard_lane_max=queue_raw.get("standard_lane_max", 2),
            deep_lane_max=queue_raw.get("deep_lane_max", 1),
            stale_expiry_minutes=queue_raw.get("stale_expiry_minutes", 5),
        ),
        phases=PhasesConfig(
            opening_minutes=phases_raw.get("opening_minutes", 5),
            wrapup_keywords=phases_raw.get("wrapup_keywords", [
                "next steps", "action items", "to summarise", "wrap up", "before we go",
            ]),
        ),
        triggers=TriggersConfig(
            client_topics=client_topics,
            consultant_hedges=triggers_raw.get("consultant_hedges", [
                "I'll get back to you",
                "I'll have to check",
                "let me verify",
                "let me confirm",
                "I'm not 100% sure",
                "that's a good question",
            ]),
        ),
        grounding=GroundingConfig(
            repo_paths=grounding_raw.get("repo_paths", [".github/instructions/"]),
            microsoft_learn=grounding_raw.get("microsoft_learn", True),
            extra_trusted_domains=grounding_raw.get("extra_trusted_domains", {}),
        ),
        output=OutputConfig(
            auto_save=output_raw.get("auto_save", True),
            include_session_summary=output_raw.get("include_session_summary", True),
        ),
        notifications=NotificationsConfig(
            sound=str(notifications_raw.get("sound", "chime")).lower(),
        ),
        speech=SpeechConfig(
            backend=backend,
            language=speech_raw.get("language", "en-GB"),
            model=speech_raw.get("model", "small.en")
                or os.environ.get("SIDEKICK_WHISPER_MODEL", "small.en"),
            compute_type=speech_raw.get("compute_type", "int8")
                or os.environ.get("SIDEKICK_WHISPER_COMPUTE", "int8"),
            capture_microphone=_as_bool(
                os.environ.get(
                    "SIDEKICK_CAPTURE_MIC",
                    speech_raw.get("capture_microphone", False),
                )
            ),
            chunk_seconds=float(speech_raw.get("chunk_seconds", 5.0)),
            vad_min_silence_ms=int(speech_raw.get("vad_min_silence_ms", 500)),
            no_speech_threshold=float(speech_raw.get("no_speech_threshold", 0.6)),
            log_prob_threshold=float(speech_raw.get("log_prob_threshold", -1.0)),
            compression_ratio_threshold=float(
                speech_raw.get("compression_ratio_threshold", 2.4)
            ),
            echo_suppression=_as_bool(speech_raw.get("echo_suppression", True)),
            speaker_naming=_as_bool(speech_raw.get("speaker_naming", True)),
        ),
        models=ModelsConfig(
            fast=models_raw.get("fast") or list(_DEFAULT_MODEL_CHAINS["fast"]),
            standard=models_raw.get("standard") or list(_DEFAULT_MODEL_CHAINS["standard"]),
            deep=models_raw.get("deep") or list(_DEFAULT_MODEL_CHAINS["deep"]),
        ),
        rules=raw.get("rules", []),
    )
