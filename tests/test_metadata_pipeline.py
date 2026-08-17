"""M2 metadata-pipeline regression tests.

Verifies that per-element metadata survives parse → split → embed → sink:
- registered parsers emitting ``list[(text, meta)]`` flow through the graph,
- ``pre_chunked`` parsers bypass the splitter,
- element mappers transform metadata (KB normalization hook),
- the ingestion report records every file that never reaches the store,
- unknown parser types are rejected at startup.

The e2e test spins up the indexer subprocess with a tmp plugin on
PYTHONPATH (the same config-driven plugin path ``serviette up`` uses) and
reads the DuckDB store to confirm per-chunk metadata landed. It requires a
Pathway license key (``PATHWAY_LICENSE_KEY``) — skipped without one, exactly
like the pre-existing ``test_indexer.py`` slow tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Registry isolation: every test starts with a clean global registry.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    from serviette.indexer import parsers as plugins

    plugins.reset_registry()
    yield
    plugins.reset_registry()


# ---------------------------------------------------------------------------
# Unit-level: ParserRegistry contract
# ---------------------------------------------------------------------------


class _FakePreChunked:
    """A registered parser whose elements bypass the splitter."""

    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        self.options = options

    def parse(self, contents: bytes, context: dict):
        return [
            ("chapter one text", {"kind": "chapter", "start_s": 0.0, "end_s": 10.0}),
            ("chapter two text", {"kind": "chapter", "start_s": 10.0, "end_s": 20.0}),
        ]


class _FakeSplittable:
    """A registered parser whose elements get re-chunked by the splitter."""

    takes_context = True
    pre_chunked = False

    def __init__(self, **options):
        self.options = options

    def parse(self, contents: bytes, context: dict):
        return [
            ("page one text " * 200, {"page_number": 1}),
            ("page two text " * 200, {"page_number": 2}),
        ]


def test_registered_parser_emits_elements_with_metadata():
    """parse() returns list[(text, meta)], not a concatenated str (M2)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert len(elements) == 2
    text0, meta0 = elements[0]
    assert text0 == "chapter one text"
    assert meta0["kind"] == "chapter"
    assert meta0["start_s"] == 0.0


def test_pre_chunked_routing():
    """pre_chunked() routes and inspects the parser's class attribute."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    assert reg.pre_chunked(".fake", "doc.fake", "/path/doc.fake") is True

    register_parser("fake_split", _FakeSplittable)
    rules = [ParserRule(match=["*.fake"], type="fake_split")]
    reg = ParserRegistry(rules)
    assert reg.pre_chunked(".fake", "doc.fake", "/path/doc.fake") is False


def test_element_mapper_transforms_metadata():
    """The element-mapper hook transforms per-element metadata."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser, register_element_mapper
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)

    def mapper(text, meta, source_meta, parser_kind):
        return {**meta, "normalized": True, "source_file": source_meta.get("name", "")}

    register_element_mapper(mapper)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert all(m.get("normalized") is True for _, m in elements)
    assert all(m["source_file"] == "doc.fake" for _, m in elements)


def test_element_mapper_none_drops_element():
    """A mapper returning None does not claim the element; with no other
    mapper registered the element is dropped WITH an ingestion-report
    record (M2 D-M2-3 — no silent drops)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser, register_element_mapper
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)

    def drop_first(text, meta, source_meta, parser_kind):
        if meta.get("start_s") == 0.0:
            return None
        return meta

    register_element_mapper(drop_first)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert len(elements) == 1
    assert elements[0][1]["start_s"] == 10.0
    drops = [r for r in reg.ingestion_report() if r["action"] == "drop_chunk"]
    assert len(drops) == 1
    assert drops[0]["stage"] == "normalize"
    assert "no element mapper claimed" in drops[0]["reason"]


def test_element_mapper_raise_skips_file():
    """A mapper raising fails the file (skip + ledger), not the pipeline."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser, register_element_mapper
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)

    def bad_mapper(text, meta, source_meta, parser_kind):
        raise ValueError("bad metadata")

    register_element_mapper(bad_mapper)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    assert reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake") == []
    report = reg.ingestion_report()
    assert len(report) == 1
    assert "normalization failure" in report[0]["reason"]


def test_skip_ledger_records_route_skips():
    """Route-skip files appear in the skip report (M2 report)."""
    from serviette.indexer.graph import ParserRegistry

    reg = ParserRegistry()
    reg.parse(b"\x00fake", ".mp4", "demo.mp4", "/path/demo.mp4")
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["file"] == "demo.mp4"
    assert report[0]["stage"] == "route"
    assert report[0]["action"] == "skip_file"
    assert "skip:" in report[0]["reason"]


class _FakeEmptyText:
    """Parser yielding one empty-text and one real element."""

    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        pass

    def parse(self, contents: bytes, context: dict):
        return [("", {"kind": "x"}), ("real text", {"kind": "x"})]


class _FakeZeroChunks:
    """Parser yielding nothing for a non-empty file."""

    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        pass

    def parse(self, contents: bytes, context: dict):
        return []


class _FakeRecording:
    """Parser emitting an ingestion-report record via context["record"]."""

    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        pass

    def parse(self, contents: bytes, context: dict):
        context["record"](
            "group", "oversize_excerpt",
            "no sentence boundary before alert threshold",
            chunk_locator="120.0-300.0s",
        )
        return [("text", {"kind": "x"})]


def test_drop_chunk_signal_keeps_file():
    """DropChunk drops one element with a record; the file keeps indexing
    (M2 D-M2-3 — skipping the whole file for one bad chunk is forbidden)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import (
        DropChunk, register_element_mapper, register_parser,
    )
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)

    def dropper(text, meta, source_meta, parser_kind):
        if meta.get("start_s") == 0.0:
            raise DropChunk("schema validation failed: page ≥ 1 required",
                            chunk_locator="0.0-10.0s")
        return meta

    register_element_mapper(dropper)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_pre")])
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert len(elements) == 1
    assert elements[0][1]["start_s"] == 10.0
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["action"] == "drop_chunk"
    assert report[0]["stage"] == "validate"
    assert report[0]["chunk_locator"] == "0.0-10.0s"
    assert "page ≥ 1" in report[0]["reason"]


def test_skip_file_signal_skips_with_reason():
    """SkipFile skips the whole file with the mapper's reason verbatim."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import (
        SkipFile, register_element_mapper, register_parser,
    )
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)

    def rejector(text, meta, source_meta, parser_kind):
        raise SkipFile("unsupported parser for KB: 'fake_pre'")

    register_element_mapper(rejector)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_pre")])
    assert reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake") == []
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["action"] == "skip_file"
    assert report[0]["stage"] == "normalize"
    assert report[0]["reason"] == "unsupported parser for KB: 'fake_pre'"


def test_empty_element_text_recorded():
    """Empty-text elements are dropped WITH a record (was a silent drop)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_empty", _FakeEmptyText)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_empty")])
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert [t for t, _ in elements] == ["real text"]
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["action"] == "drop_chunk"
    assert report[0]["stage"] == "parse"
    assert report[0]["reason"] == "empty element text"


def test_zero_chunks_red_flag():
    """A non-empty file producing no chunks and no other record is a red
    flag (M2 D-M2-4)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_zero", _FakeZeroChunks)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_zero")])
    assert reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake") == []
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["action"] == "zero_chunks"
    assert report[0]["stage"] == "parse"


def test_parse_failure_does_not_double_record():
    """A recorded parse failure must not also get a zero_chunks record."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    class _Boom:
        takes_context = True

        def __init__(self, **options):
            pass

        def parse(self, contents, context):
            raise RuntimeError("corrupt file")

    register_parser("fake_boom", _Boom)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_boom")])
    assert reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake") == []
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["action"] == "skip_file"
    assert report[0]["stage"] == "parse"


def test_context_record_callback():
    """takes_context parsers emit records via context['record'] (M2:
    oversize_excerpt from the vidprep transcript splitter)."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_rec", _FakeRecording)
    reg = ParserRegistry([ParserRule(match=["*.fake"], type="fake_rec")])
    elements = reg.parse(b"contents", ".fake", "doc.fake", "/path/doc.fake")
    assert len(elements) == 1
    report = reg.ingestion_report()
    assert len(report) == 1
    assert report[0]["stage"] == "group"
    assert report[0]["action"] == "oversize_excerpt"
    assert report[0]["parser"] == "fake_rec"
    assert report[0]["chunk_locator"] == "120.0-300.0s"


def test_report_writer_files(tmp_path):
    """The on-disk report mirrors records: events.jsonl + summary.txt
    with matching counts (M2 §Error recording)."""
    from serviette.indexer.graph import (
        ParserRegistry, _IngestionReportWriter,
    )

    reg = ParserRegistry()
    reg.report_writer = _IngestionReportWriter(tmp_path)
    reg.parse(b"\x00fake", ".mp4", "demo.mp4", "/path/demo.mp4")
    reg.parse(b"\x00fake", ".mkv", "other.mkv", "/path/other.mkv")

    run_dirs = list(tmp_path.glob("ingestion_*"))
    assert len(run_dirs) == 1
    events = (run_dirs[0] / "events.jsonl").read_text().strip().splitlines()
    assert len(events) == 2
    parsed = [json.loads(line) for line in events]
    assert {r["file"] for r in parsed} == {"demo.mp4", "other.mkv"}
    assert all(r["stage"] == "route" and r["action"] == "skip_file"
               for r in parsed)
    assert all("ts" in r for r in parsed)
    summary = (run_dirs[0] / "summary.txt").read_text()
    assert "route/skip_file: 2" in summary
    assert "demo.mp4" in summary and "other.mkv" in summary


def test_unknown_parser_type_rejected_at_startup():
    """check_rule_deps rejects names that are neither built-in, registered, nor skip."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.config.schema import ParserRule

    rules = [ParserRule(match=["*.fake"], type="nonexistent_parser")]
    reg = ParserRegistry(rules)
    with pytest.raises(ValueError, match="unknown"):
        reg.check_rule_deps()


def test_registered_parser_accepted_by_check_rule_deps():
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)
    rules = [ParserRule(match=["*.fake"], type="fake_pre")]
    reg = ParserRegistry(rules)
    reg.check_rule_deps()  # must not raise


def test_path_aware_routing():
    """Rules match against the full path (*.vidprep/bundle.yaml), not just basename."""
    from serviette.indexer.graph import ParserRegistry
    from serviette.indexer.parsers import register_parser
    from serviette.config.schema import ParserRule

    register_parser("fake_pre", _FakePreChunked)
    rules = [ParserRule(match=["*.vidprep/bundle.yaml"], type="fake_pre")]
    reg = ParserRegistry(rules)
    # basename "bundle.yaml" alone doesn't match; the full path does.
    kind, _ = reg._route(".yaml", "bundle.yaml", "/data/X.mp4.vidprep/bundle.yaml")
    assert kind == "fake_pre"


# ---------------------------------------------------------------------------
# E2e: subprocess indexer with a tmp plugin, metadata in DuckDB
# ---------------------------------------------------------------------------


_HAS_LICENSE = bool(os.environ.get("PATHWAY_LICENSE_KEY"))


def _write_plugin(tmp_path: Path) -> str:
    """Write a fake parser plugin module to tmp_path; return its module name."""
    plugin = tmp_path / "fake_plugin.py"
    plugin.write_text(
        '''
from serviette.indexer.parsers import register_parser


class FakeParser:
    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        pass

    def parse(self, contents, context):
        return [
            ("alpha content", {"artifact": "stitched", "kind": "chapter",
             "start_s": 0.0, "end_s": 10.0}),
            ("beta content", {"artifact": "stitched", "kind": "chapter",
             "start_s": 10.0, "end_s": 20.0}),
        ]


def register(config):
    register_parser("fakeparse", FakeParser)
'''
    )
    return "fake_plugin"


def _write_config(tmp_path: Path, docs_dir: Path, store: Path) -> Path:
    config = {
        "sources": [
            {"type": "fs", "path": str(docs_dir), "glob": "*.fake", "mode": "static"}
        ],
        "vector_db": {"type": "duckdb", "path": str(store), "table": "chunks"},
        "embedder": {"type": "mock"},
        "splitter": {"type": "token_count", "chunk_size": 512, "chunk_overlap": 50},
        "persistence": {"enabled": False},
        "parser_plugins": [_write_plugin(tmp_path)],
        "parser": [
            {"match": ["*.fake"], "type": "fakeparse"},
        ],
    }
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return cfg_path


@pytest.mark.skipif(not _HAS_LICENSE, reason="PATHWAY_LICENSE_KEY not set")
def test_metadata_survives_to_duckdb(tmp_path):
    """E2e: per-element metadata from a registered plugin lands in DuckDB."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc1.fake").write_text("placeholder")

    store = tmp_path / "store.duckdb"
    cfg_path = _write_config(tmp_path, docs, store)

    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, "-m", "serviette.cli", "indexer", "--config", str(cfg_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"indexer failed:\n{proc.stderr[-2000:]}"

    import duckdb

    conn = duckdb.connect(str(store), read_only=True)
    try:
        rows = conn.execute("SELECT text, metadata FROM chunks").fetchall()
    finally:
        conn.close()

    assert len(rows) == 2  # two pre-chunked elements, no splitting
    texts = {r[0] for r in rows}
    assert texts == {"alpha content", "beta content"}
    metas = [json.loads(r[1]) for r in rows]
    # Per-element metadata survived (merged with source _metadata).
    by_text = {m.get("start_s"): m for m in metas}
    assert by_text[0.0]["artifact"] == "stitched"
    assert by_text[0.0]["kind"] == "chapter"
    assert by_text[10.0]["start_s"] == 10.0
    assert by_text[10.0]["end_s"] == 20.0
    # Source _metadata is merged in (path from the fs connector).
    assert all("path" in m for m in metas)


def _write_mapper_plugin(tmp_path: Path) -> str:
    """Plugin that registers both a parser AND an element mapper producing a
    strict envelope. Used to assert the mapper-active path stores the
    envelope verbatim (no fs connector keys leak)."""
    plugin = tmp_path / "mapper_plugin.py"
    plugin.write_text(
        '''
from serviette.indexer.parsers import register_parser, register_element_mapper


class FakeParser:
    takes_context = True
    pre_chunked = True

    def __init__(self, **options):
        pass

    def parse(self, contents, context):
        return [
            ("alpha", {"kind": "chapter", "artifact": "stitched",
             "start_s": 0.0, "end_s": 10.0, "title": "A"}),
            ("beta", {"kind": "chapter", "artifact": "stitched",
             "start_s": 10.0, "end_s": 20.0, "title": "B"}),
        ]


_ENVELOPE_KEYS = {
    "kind", "artifact", "start_s", "end_s", "title",
    "schema_version", "curriculum", "session", "source_file",
    "source_kind", "extraction",
}


def mapper(text, meta, source_meta, parser_kind):
    return {
        "schema_version": 1,
        "curriculum": "testcourse",
        "session": "testsession",
        "source_file": source_meta.get("name", ""),
        "source_kind": "video",
        "extraction": "test-v1",
        **meta,
    }


def register(config):
    register_parser("mapperparse", FakeParser)
    register_element_mapper(mapper)
'''
    )
    return "mapper_plugin"


@pytest.mark.skipif(not _HAS_LICENSE, reason="PATHWAY_LICENSE_KEY not set")
def test_mapper_active_stores_envelope_verbatim(tmp_path):
    """E2e: when a mapper is registered, stored metadata = mapper output
    exactly — no fs connector keys (path/owner/size/mtime/…) leak on top of
    the validated envelope (additionalProperties: false freeze)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc1.fake").write_text("placeholder")

    store = tmp_path / "store.duckdb"
    config = {
        "sources": [
            {"type": "fs", "path": str(docs), "glob": "*.fake", "mode": "static"}
        ],
        "vector_db": {"type": "duckdb", "path": str(store), "table": "chunks"},
        "embedder": {"type": "mock"},
        "splitter": {"type": "token_count", "chunk_size": 512, "chunk_overlap": 50},
        "persistence": {"enabled": False},
        "parser_plugins": [_write_mapper_plugin(tmp_path)],
        "parser": [
            {"match": ["*.fake"], "type": "mapperparse"},
        ],
    }
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    env = dict(
        os.environ,
        PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    proc = subprocess.run(
        [sys.executable, "-m", "serviette.cli", "indexer", "--config", str(cfg_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"indexer failed:\n{proc.stderr[-2000:]}"

    import duckdb

    conn = duckdb.connect(str(store), read_only=True)
    try:
        rows = conn.execute("SELECT text, metadata FROM chunks").fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    for text, meta_json in rows:
        meta = json.loads(meta_json)
        # The envelope only — no fs connector keys leaked (the freeze-break
        # this guards against: additionalProperties: false).
        assert "path" not in meta, f"fs 'path' leaked into envelope: {meta}"
        assert "owner" not in meta, f"fs 'owner' leaked into envelope: {meta}"
        assert "size" not in meta, f"fs 'size' leaked into envelope: {meta}"
        assert "mtime" not in meta, f"fs 'mtime' leaked: {meta}"
        # Mapper output verbatim: every mapper field present.
        assert meta["curriculum"] == "testcourse"
        assert meta["session"] == "testsession"
        assert meta["source_kind"] == "video"
        assert meta["extraction"] == "test-v1"
        assert meta["kind"] == "chapter"
        assert meta["artifact"] == "stitched"
        assert "start_s" in meta and "end_s" in meta
