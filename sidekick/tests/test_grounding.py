"""Characterization tests for the extracted grounding builder (Phase 2c).

These pin the existing behaviour of ``build_grounding_context`` so the
structural extraction from ``server._build_grounding_context`` is verifiably
behaviour-preserving.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sidekick import grounding


@dataclass
class _FakeGrounding:
    repo_paths: list[str] = field(default_factory=list)


@dataclass
class _FakeConfig:
    domains: list[str] = field(default_factory=list)
    repo_paths: list[str] = field(default_factory=list)
    customer: str = "default"

    @property
    def grounding(self):
        return _FakeGrounding(repo_paths=self.repo_paths)


@dataclass
class _FakeContext:
    context_documents: list[str] = field(default_factory=list)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the builder at a tmp workspace + tmp home directory."""
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    monkeypatch.setenv("SIDEKICK_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return workspace, home


class TestNoConfig:
    def test_none_config_returns_placeholder(self):
        assert grounding.build_grounding_context(None, None) == "(no config loaded)"


class TestEmpty:
    def test_nothing_to_ground_on(self, isolated):
        cfg = _FakeConfig(domains=[], repo_paths=[], customer="acme")
        result = grounding.build_grounding_context(cfg, None)
        assert result == "(no grounding context available)"


class TestInstructionFiles:
    def test_domain_keyword_loads_instruction_file(self, isolated):
        workspace, _ = isolated
        instr = workspace / ".github" / "instructions"
        instr.mkdir(parents=True)
        (instr / "pyspark-notebooks.instructions.md").write_text(
            "Use Delta MERGE for idempotent loads.", encoding="utf-8"
        )
        cfg = _FakeConfig(domains=["PySpark notebooks"], repo_paths=[])
        result = grounding.build_grounding_context(cfg, None)
        assert "--- pyspark-notebooks standards ---" in result
        assert "Delta MERGE" in result

    def test_instruction_content_truncated_to_800(self, isolated):
        workspace, _ = isolated
        instr = workspace / ".github" / "instructions"
        instr.mkdir(parents=True)
        (instr / "dax-powerbi.instructions.md").write_text(
            "x" * 2000, encoding="utf-8"
        )
        cfg = _FakeConfig(domains=["DAX"], repo_paths=[])
        result = grounding.build_grounding_context(cfg, None)
        body = result.split("standards ---\n", 1)[1]
        assert len(body) == 800

    def test_each_file_loaded_once(self, isolated):
        workspace, _ = isolated
        instr = workspace / ".github" / "instructions"
        instr.mkdir(parents=True)
        (instr / "dax-powerbi.instructions.md").write_text("dax", encoding="utf-8")
        # Two keywords ("dax", "power bi") both map to dax-powerbi
        cfg = _FakeConfig(domains=["dax", "power bi"], repo_paths=[])
        result = grounding.build_grounding_context(cfg, None)
        assert result.count("--- dax-powerbi standards ---") == 1


class TestArtifacts:
    def test_loads_top_3_recent_artifacts_newest_first(self, isolated):
        workspace, _ = isolated
        repo = workspace / "engagement"
        repo.mkdir()
        # Create 4 files with increasing mtimes
        for i in range(4):
            f = repo / f"doc{i}.md"
            f.write_text(f"content {i}", encoding="utf-8")
            import os

            os.utime(f, (1000 + i, 1000 + i))
        cfg = _FakeConfig(domains=[], repo_paths=["engagement"])
        result = grounding.build_grounding_context(cfg, None)
        # Newest 3 (doc3, doc2, doc1) present; oldest (doc0) excluded
        assert "content 3" in result
        assert "content 2" in result
        assert "content 1" in result
        assert "content 0" not in result

    def test_artifact_content_truncated_to_1200(self, isolated):
        workspace, _ = isolated
        repo = workspace / "engagement"
        repo.mkdir()
        (repo / "big.md").write_text("y" * 3000, encoding="utf-8")
        cfg = _FakeConfig(domains=[], repo_paths=["engagement"])
        result = grounding.build_grounding_context(cfg, None)
        body = result.split("(recent artifact) ---\n", 1)[1]
        assert len(body) == 1200

    def test_instructions_repo_path_skipped(self, isolated):
        workspace, _ = isolated
        instr = workspace / ".github" / "instructions"
        instr.mkdir(parents=True)
        (instr / "note.md").write_text("should not appear as artifact", encoding="utf-8")
        cfg = _FakeConfig(domains=[], repo_paths=[".github/instructions"])
        result = grounding.build_grounding_context(cfg, None)
        assert "recent artifact" not in result


class TestSessionSummaries:
    def test_loads_top_2_summaries(self, isolated):
        _, home = isolated
        out = home / ".sidekick" / "outputs" / "acme"
        out.mkdir(parents=True)
        import os

        for i in range(3):
            f = out / f"sidekick_summary_{i}.md"
            f.write_text(f"summary {i}", encoding="utf-8")
            os.utime(f, (2000 + i, 2000 + i))
        cfg = _FakeConfig(domains=[], repo_paths=[], customer="acme")
        result = grounding.build_grounding_context(cfg, None)
        assert "Previous session: sidekick_summary_2.md" in result
        assert "Previous session: sidekick_summary_1.md" in result
        assert "sidekick_summary_0.md" not in result


class TestLiveContext:
    def test_injects_last_5_documents(self, isolated):
        ctx = _FakeContext(context_documents=[f"doc{i}" for i in range(7)])
        cfg = _FakeConfig(domains=[], repo_paths=[])
        result = grounding.build_grounding_context(cfg, ctx)
        # Last 5 (doc2..doc6) present, first 2 excluded
        assert "doc6" in result
        assert "doc2" in result
        assert "--- Live context #1 ---" in result
        assert "doc0" not in result

    def test_live_context_truncated_to_1500(self, isolated):
        ctx = _FakeContext(context_documents=["z" * 3000])
        cfg = _FakeConfig(domains=[], repo_paths=[])
        result = grounding.build_grounding_context(cfg, ctx)
        body = result.split("Live context #1 ---\n", 1)[1]
        assert len(body) == 1500
