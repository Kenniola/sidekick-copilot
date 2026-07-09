"""Session log — records outputs and generates meeting summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sidekick.config import SidekickConfig, get_output_dir

logger = logging.getLogger(__name__)


class SessionLog:
    """Records Sidekick outputs during a meeting session."""

    def __init__(self, config: SidekickConfig):
        self.config = config
        self.outputs: list[dict] = []
        self.session_start = datetime.now(timezone.utc)

    def record(self, result) -> None:
        """Record an action result."""
        self.outputs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": getattr(result, "action_type", "unknown"),
            "question": getattr(result, "question", ""),
            "answer": getattr(result, "answer", ""),
            "sources": getattr(result, "sources", []),
            "confidence": getattr(result, "confidence", "medium"),
        })

    def generate_summary(self, context) -> str:
        """Generate a structured meeting summary."""
        duration = (
            datetime.now(timezone.utc) - self.session_start
        ).total_seconds() / 60.0

        parts = [
            f"━━━ SESSION SUMMARY — {self.config.customer} ━━━",
            f"Duration: {duration:.0f} minutes",
            f"Outputs generated: {len(self.outputs)}",
            "",
        ]

        # Group outputs by type
        by_type: dict[str, list] = {}
        for o in self.outputs:
            by_type.setdefault(o["action_type"], []).append(o)

        for action_type, items in by_type.items():
            parts.append(f"[{action_type.upper()}] ({len(items)})")
            for item in items:
                parts.append(f"  • {item['question'][:80]}")
            parts.append("")

        # Open threads
        if context and hasattr(context, "threads"):
            open_threads = [
                t for t in context.threads.values()
                if t.status in ("open", "blocked")
            ]
            if open_threads:
                parts.append("OPEN THREADS:")
                for t in open_threads:
                    parts.append(f"  ⏳ {t.topic}")
                parts.append("")

        # Action items
        if context and hasattr(context, "action_items") and context.action_items:
            parts.append("ACTION ITEMS:")
            for ai in context.action_items:
                parts.append(f"  □ {ai.get('description', str(ai))}")
            parts.append("")

        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(parts)

    def format_outputs(self) -> str:
        """Format recent outputs for status display."""
        if not self.outputs:
            return "  (none yet)"
        recent = self.outputs[-5:]
        return "\n".join(
            f"  [{o['action_type']}] {o['question'][:60]}" for o in recent
        )

    def save_to_disk(self) -> Path | None:
        """Save session outputs to the user's output directory."""
        if not self.config.output.auto_save:
            return None

        output_dir = get_output_dir(self.config.customer)

        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"sidekick_session_{timestamp}.json"
        output_path = output_dir / filename

        output_path.write_text(
            json.dumps(
                {
                    "customer": self.config.customer,
                    "session_start": self.session_start.isoformat(),
                    "session_end": datetime.now(timezone.utc).isoformat(),
                    "outputs": self.outputs,
                },
                indent=2,
            )
        )

        logger.info("Session saved to %s", output_path)
        return output_path

    def save_transcript(self, context) -> Path | None:
        """Save the full transcript as a plain-text file."""
        if not context or not hasattr(context, "full_transcript"):
            return None

        transcript = getattr(context, "full_transcript", [])
        if not transcript:
            return None

        output_dir = get_output_dir(self.config.customer)
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"transcript_{timestamp}.txt"

        lines = []
        for line in transcript:
            ts = getattr(line, "start", "?")
            speaker = getattr(line, "speaker", "?")
            text = getattr(line, "text", str(line))
            lines.append(f"[{ts}] {speaker}: {text}")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Transcript saved to %s", path)
        return path

    def save_markdown_summary(self, context) -> Path | None:
        """Save a structured markdown meeting summary."""
        if not context:
            return None

        output_dir = get_output_dir(self.config.customer)
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"summary_{timestamp}.md"

        duration = (
            datetime.now(timezone.utc) - self.session_start
        ).total_seconds() / 60.0

        parts = [
            f"# Meeting Summary — {self.config.customer}",
            "",
            f"**Date**: {self.session_start.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Duration**: {duration:.0f} minutes",
            f"**Outputs generated**: {len(self.outputs)}",
            "",
        ]

        # Key topics from threads
        if hasattr(context, "threads") and context.threads:
            parts.append("## Topics Discussed")
            parts.append("")
            for t in context.threads.values():
                status_icon = {"open": "🔵", "closed": "✅", "blocked": "🔴"}.get(
                    t.status, "⚪"
                )
                parts.append(f"- {status_icon} **{t.topic}** ({t.status})")
                for fact in getattr(t, "key_facts", [])[:3]:
                    parts.append(f"  - {fact}")
            parts.append("")

        # Research results
        research = [o for o in self.outputs if o["action_type"] == "research"]
        if research:
            parts.append("## Research Answers")
            parts.append("")
            for r in research:
                parts.append(f"### {r['question']}")
                parts.append("")
                parts.append(r["answer"][:500])
                parts.append("")

        # Prototypes
        prototypes = [o for o in self.outputs if o["action_type"] == "prototype"]
        if prototypes:
            parts.append("## Code Prototypes")
            parts.append("")
            for p in prototypes:
                parts.append(f"- {p['question'][:80]}")
            parts.append("")

        # Action items
        if hasattr(context, "action_items") and context.action_items:
            parts.append("## Action Items")
            parts.append("")
            for ai in context.action_items:
                parts.append(f"- [ ] {ai.get('description', str(ai))}")
            parts.append("")

        # Open threads
        if hasattr(context, "threads"):
            open_threads = [
                t for t in context.threads.values()
                if t.status in ("open", "blocked")
            ]
            if open_threads:
                parts.append("## Open Items for Follow-up")
                parts.append("")
                for t in open_threads:
                    parts.append(f"- {t.topic}")
                parts.append("")

        path.write_text("\n".join(parts), encoding="utf-8")
        logger.info("Markdown summary saved to %s", path)
        return path
