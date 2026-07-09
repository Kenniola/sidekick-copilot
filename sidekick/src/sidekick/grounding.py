"""Grounding context builder — team standards + recent engagement artifacts.

Extracted from ``server._build_grounding_context`` (Phase 2c). This is pure,
synchronous file I/O that takes the loaded config and live meeting context
explicitly (rather than reaching for module globals), so it can be tested in
isolation. Callers should wrap it in ``asyncio.to_thread`` to avoid blocking
the event loop.
"""

from __future__ import annotations

import os
from pathlib import Path

# Map domain keywords to the matching .github/instructions file stem.
_KEYWORD_TO_FILE: dict[str, str] = {
    "pyspark": "pyspark-notebooks",
    "notebook": "pyspark-notebooks",
    "spark": "pyspark-notebooks",
    "warehouse": "tsql-warehouse",
    "sql": "tsql-warehouse",
    "t-sql": "tsql-warehouse",
    "dax": "dax-powerbi",
    "power bi": "dax-powerbi",
    "powerbi": "dax-powerbi",
    "semantic model": "dax-powerbi",
    "directlake": "dax-powerbi",
    "dataflow": "dataflows-pipelines",
    "pipeline": "dataflows-pipelines",
    "governance": "governance-security",
    "purview": "governance-security",
    "security": "governance-security",
    "rls": "governance-security",
    "aws": "cross-cloud-integration",
    "s3": "cross-cloud-integration",
    "cross-cloud": "cross-cloud-integration",
}


def build_grounding_context(config, context) -> str:
    """Build grounding context from instruction files and engagement artifacts.

    Loads team standards from ``.github/instructions/`` (matched to the config's
    domains), recent engagement artifacts from configured repo paths, previous
    session summaries for the customer, and any live injected context.

    Args:
        config: the loaded ``SidekickConfig`` (or ``None`` if not yet loaded).
        context: the live ``MeetingContext`` (or ``None``).

    Returns:
        A newline-joined grounding block, or a placeholder string when there is
        no config / nothing to ground on.
    """
    if not config:
        return "(no config loaded)"

    workspace_root = Path(os.environ.get("SIDEKICK_WORKSPACE_ROOT", "."))
    parts: list[str] = []

    # 1. Load relevant instruction files based on configured domains
    instructions_dir = workspace_root / ".github" / "instructions"
    if instructions_dir.exists():
        domain_keywords = [d.lower() for d in config.domains]
        loaded_files: set[str] = set()
        for domain in domain_keywords:
            for kw, fname in _KEYWORD_TO_FILE.items():
                if kw in domain and fname not in loaded_files:
                    fpath = instructions_dir / f"{fname}.instructions.md"
                    if fpath.exists():
                        try:
                            content = fpath.read_text(encoding="utf-8")
                            # Take first 800 chars to stay within context limits
                            parts.append(f"--- {fname} standards ---\n{content[:800]}")
                            loaded_files.add(fname)
                        except Exception:
                            pass

    # 2. Load recent engagement artifacts (meeting preps, QA summaries)
    for repo_path_str in config.grounding.repo_paths:
        repo_path = workspace_root / repo_path_str
        if not repo_path.exists():
            continue
        # Skip the instructions directory (already loaded above)
        if repo_path_str.rstrip("/").endswith("instructions"):
            continue

        # Search for recent meeting prep and summary files
        artifact_files: list[tuple[float, Path]] = []
        for suffix in ("*.md", "*.txt"):
            for f in repo_path.rglob(suffix):
                try:
                    artifact_files.append((f.stat().st_mtime, f))
                except Exception:
                    continue

        # Sort by modification time (newest first), take top 3
        artifact_files.sort(key=lambda x: x[0], reverse=True)
        for _, f in artifact_files[:3]:
            try:
                content = f.read_text(encoding="utf-8")
                rel = f.relative_to(workspace_root)
                parts.append(f"--- {rel} (recent artifact) ---\n{content[:1200]}")
            except Exception:
                continue

    # 3. Load previous session summaries for this customer
    outputs_dir = Path.home() / ".sidekick" / "outputs" / (config.customer or "default")
    if outputs_dir.exists():
        summary_files = sorted(
            outputs_dir.glob("sidekick_summary_*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for sf in summary_files[:2]:
            try:
                content = sf.read_text(encoding="utf-8")
                parts.append(f"--- Previous session: {sf.name} ---\n{content[:400]}")
            except Exception:
                continue

    # 4. Injected live context (from add_context tool)
    if context and context.context_documents:
        for i, doc in enumerate(context.context_documents[-5:], 1):
            parts.append(f"--- Live context #{i} ---\n{doc[:1500]}")

    return "\n\n".join(parts) if parts else "(no grounding context available)"
