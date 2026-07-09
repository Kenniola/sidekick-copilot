"""Prototype pipeline — generates working code skeletons."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sidekick.config import SidekickConfig
from sidekick.llm import call_llm

logger = logging.getLogger(__name__)

PROTOTYPE_SYSTEM_PROMPT = """You are a Microsoft Fabric code generator. Generate \
working code prototypes following the team's standards.

CODE STANDARDS:
- PySpark: medallion architecture, Delta format, audit columns, explicit schemas
- T-SQL: uppercase keywords, lowercase identifiers, explicit JOINs, alias tables
- DAX: one function per line, indent arguments, use VAR for intermediates
- No SELECT *, no hardcoded values, always parameterised

OUTPUT:
- Return ONLY the code block with language tag
- Include brief inline comments explaining key decisions
- Add TODO markers for values that need customer-specific configuration"""


@dataclass
class PrototypeResult:
    """Result from the prototype pipeline."""

    description: str
    prototype_type: str
    code: str = ""
    language: str = "python"

    def format(self) -> str:
        return f"""```{self.language}
{self.code}
```

Type: {self.prototype_type} | Generated from meeting context"""


class PrototypePipeline:
    """Generate working code prototypes from meeting context."""

    def __init__(self, config: SidekickConfig | None = None):
        self.config = config

    async def execute_direct(
        self,
        description: str,
        prototype_type: str = "notebook",
        columns: str = "",
        context=None,
    ) -> PrototypeResult:
        """Generate a code prototype directly."""
        language = {
            "notebook": "python",
            "sql": "sql",
            "dax": "dax",
            "pipeline": "python",
        }.get(prototype_type, "python")

        user_prompt = f"""PROTOTYPE REQUEST:
Type: {prototype_type}
Description: {description}
{f'Columns: {columns}' if columns else ''}

MEETING CONTEXT:
{self._format_context(context)}

CUSTOMER: {self.config.customer if self.config else 'General'}

Generate a working {prototype_type} prototype."""

        code = await call_llm(
            system_prompt=PROTOTYPE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tier="deep",
        )

        # Strip markdown fences if the LLM wrapped the output
        code = self._strip_fences(code)

        return PrototypeResult(
            description=description,
            prototype_type=prototype_type,
            code=code,
            language=language,
        )

    def _format_context(self, context) -> str:
        if not context:
            return "(no meeting context)"
        facts = getattr(context, "key_facts", [])
        return "\n".join(f"- {f}" for f in facts) if facts else "(no key facts)"

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output."""
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
