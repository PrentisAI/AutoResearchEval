"""Offline checks: import hygiene, packaged data, taxonomy surface, QA gates.

Nothing here touches the network or spawns a session, so it runs in CI. The two calls
that do (generate_analysis, label_arft) are exercised by the end-to-end harness against
a stub CLI and a local mock endpoint.
"""
import json
import os
import subprocess
import sys

import pytest

import autoresearcheval as aae
from autoresearcheval import analysis_qa, config, label_qa, patterns, traj_tools


def test_public_surface():
    assert aae.__all__ == ["generate_analysis", "label_arft", "pattern_info",
                           "all_patterns", "__version__"]
    assert callable(aae.generate_analysis) and callable(aae.label_arft)


def test_import_has_no_side_effects(tmp_path):
    """Importing must not read the environment or scan the cwd.

    The pre-package code resolved the corpus root and globbed it for model directories
    at import time, so importing the classifier from the wrong directory changed its
    behaviour. Guard that it stays fixed: import in a clean subprocess, in an empty cwd,
    with the corpus env var pointing somewhere that does not exist.
    """
    env = {**os.environ, "AAJ_CORPUS_DIR": str(tmp_path / "nope")}
    code = "import autoresearcheval; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_packaged_data_ships():
    for path in (config.onboarding(), config.exemplar(), config.arft_guide()):
        assert path.exists(), f"{path} missing from the wheel"
        assert path.stat().st_size > 1000


def test_models_empty_without_a_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("AAJ_CORPUS_DIR", str(tmp_path / "absent"))
    assert config.models() == []


def test_env_overrides_are_read_at_call_time(tmp_path, monkeypatch):
    custom = tmp_path / "mine.md"
    custom.write_text("# my exemplar\n")
    monkeypatch.setenv("AAJ_EXEMPLAR", str(custom))
    assert config.exemplar() == custom


@pytest.mark.parametrize("code", ["A.1", "C.1", "F.4", "X.8"])
def test_pattern_info(code):
    info = aae.pattern_info(code)
    assert info["code"] == code
    assert info["stage"] == code.split(".")[0]
    assert info["root_cause"] in ("grounding", "depth", "integrity", "robustness")


def test_taxonomy_is_complete_and_consistent():
    everything = aae.all_patterns()
    assert len(everything) == 45
    # every pattern maps to exactly one pillar, and every pillar word is reachable
    words = {i["root_cause"] for i in everything.values()}
    assert words == {"grounding", "depth", "integrity", "robustness"}
    assert set(patterns.STAGES) == set("ABCDEFX")


def test_pattern_info_rejects_unknown_codes():
    with pytest.raises(KeyError):
        aae.pattern_info("Z.9")


def test_shipped_exemplar_passes_its_own_quality_gate():
    """The exemplar is handed to every session as the bar; it has to clear the bar."""
    report = analysis_qa.check(config.exemplar(), "soft[current_density]")
    assert report["ok"], report["problems"]
    assert report["issues"] >= 28


def test_label_qa_rejects_credit_only_evidence(tmp_path):
    """Polarity guard: a label whose only evidence is the fair-credit section."""
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({
        "task_id": "t", "overall_severity": "high", "summary": "s",
        "hits": [{"code": "C.2", "confidence": 0.8, "evidence": "## Credit Due",
                  "why": "credit-only"}],
        "partials": []}))
    res = label_qa.check(bad)
    assert not res["ok"]
    assert any("Credit Due" in p for p in res["problems"])


def test_label_qa_accepts_a_well_formed_classification(tmp_path):
    good = tmp_path / "c.json"
    good.write_text(json.dumps({
        "task_id": "t", "overall_severity": "high", "summary": "s",
        "hits": [{"code": "C.3", "confidence": 0.9,
                  "evidence": "Section C, issue 4: the delivered script differs",
                  "why": "matches C.3"}],
        "partials": []}))
    assert label_qa.check(good)["ok"]


def test_traj_tools_detects_each_supported_format():
    assert traj_tools.detect_format({"claude_log": "{}"}) == "claude"
    assert traj_tools.detect_format({"gemini_log": "x"}) == "gemini"
    assert traj_tools.detect_format({"codex_log": "x"}) == "codex"


def test_traj_tools_replays_edits_onto_the_written_file(tmp_path):
    """Iron rule 1: judge the final delivered artifact, not an intermediate draft."""
    log = tmp_path / "claude_log.jsonl"
    log.write_text("\n".join(json.dumps(o) for o in [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "1", "name": "Write",
             "input": {"file_path": "/w/m.py", "content": "x = 1\n"}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "2", "name": "Edit",
             "input": {"file_path": "/w/m.py", "old_string": "x = 1",
                       "new_string": "x = 2"}}]}},
    ]))
    assert traj_tools.reconstruct_file(str(log), "m.py").strip() == "x = 2"


def test_harness_notes_have_usable_defaults():
    """The defaults must be instructions, not placeholders.

    They are quoted verbatim into every session, so a leftover TODO would become part
    of the prompt. They also must not assert anything about the harness: telling an
    analyst that a real search tool is mocked manufactures fabrication findings, and
    the reverse credits calibration against literature never fetched.
    """
    from autoresearcheval import generate as gen
    for note in (gen.RETRIEVAL_NOTE, gen.GOLD_NOTE):
        assert "TODO" not in note
        assert len(note) > 200
    assert "has not been declared" in gen.RETRIEVAL_NOTE
    assert "do not" in gen.RETRIEVAL_NOTE.lower()
    assert "unless you actually" in gen.GOLD_NOTE


def test_notes_reach_the_prompt():
    from autoresearcheval import generate as gen
    t = {"task_id": "t1", "reward": 0.0, "reason": "soft[x]", "category": "soft"}
    prompt = gen.build_instruction(t, "m", "/tmp/out")
    assert gen.RETRIEVAL_NOTE in prompt and gen.GOLD_NOTE in prompt


def test_trajectory_accepts_dict_file_and_directory(tmp_path):
    from autoresearcheval.api import _resolve_trajectory

    record, path = _resolve_trajectory({"task_id": "inline", "claude_log": "{}"})
    assert record["task_id"] == "inline" and path is None

    one = tmp_path / "solo"
    (one / "traj").mkdir(parents=True)
    (one / "traj" / "a.json").write_text(json.dumps({"task_id": "a", "claude_log": "{}"}))
    record, path = _resolve_trajectory(one)
    assert record["task_id"] == "a" and path.name == "a.json"

    record, path = _resolve_trajectory(one / "traj" / "a.json")
    assert record["task_id"] == "a"


def test_ambiguous_directory_points_at_the_batch_cli(tmp_path):
    """Picking one of several arbitrarily would analyze a trajectory nobody named."""
    many = tmp_path / "many"
    (many / "traj").mkdir(parents=True)
    for n in "ab":
        (many / "traj" / f"{n}.json").write_text(json.dumps({"task_id": n, "claude_log": "{}"}))
    from autoresearcheval.api import _resolve_trajectory
    with pytest.raises(ValueError, match="aaj-generate --run-dir"):
        _resolve_trajectory(many)


def test_empty_directory_says_where_it_looked(tmp_path):
    from autoresearcheval.api import _resolve_trajectory
    with pytest.raises(FileNotFoundError, match="no trajectory JSON"):
        _resolve_trajectory(tmp_path)
