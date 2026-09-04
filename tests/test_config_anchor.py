"""Config-file-relative path anchoring in ``load_config`` (spec D5).

Relative filesystem paths (DuckDB store file, persistence directory,
ingestion log directory, ``fs`` source paths) resolve against the config
file's directory, regardless of the process working directory. Absolute
and ``~``-prefixed paths pass through (the latter expanded); non-duckdb
backends, non-fs sources and disabled persistence are untouched;
``load_config_dict`` keeps relative paths verbatim.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from serviette.config.schema import load_config, load_config_dict

CONFIG = {
    "sources": [{"type": "fs", "path": "docs"}],
    "vector_db": {"type": "duckdb", "path": "store.duckdb"},
    "persistence": {"enabled": True, "path": "state"},
    "indexer": {"ingestion_log_dir": "log"},
}


def _write(tmp_path: Path, data: dict) -> Path:
    sub = tmp_path / "sub"
    sub.mkdir(exist_ok=True)
    path = sub / "cfg.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _copy(data: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(data))


def test_relative_paths_anchor_to_config_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd deliberately != config dir
    cfg = load_config(_write(tmp_path, CONFIG))
    assert Path(cfg.sources[0].path) == tmp_path / "sub" / "docs"
    assert Path(cfg.vector_db.path) == tmp_path / "sub" / "store.duckdb"
    assert Path(cfg.persistence.path) == tmp_path / "sub" / "state"
    assert Path(cfg.indexer.ingestion_log_dir) == tmp_path / "sub" / "log"


def test_absolute_paths_untouched(tmp_path):
    data = _copy(CONFIG)
    data["vector_db"]["path"] = str(tmp_path / "abs.duckdb")
    data["sources"][0]["path"] = str(tmp_path / "abs-docs")
    cfg = load_config(_write(tmp_path, data))
    assert Path(cfg.vector_db.path) == tmp_path / "abs.duckdb"
    assert Path(cfg.sources[0].path) == tmp_path / "abs-docs"


def test_tilde_paths_expand_but_not_rebase(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    data = _copy(CONFIG)
    data["vector_db"]["path"] = "~/store.duckdb"
    cfg = load_config(_write(tmp_path, data))
    assert Path(cfg.vector_db.path) == tmp_path / "home" / "store.duckdb"


def test_non_duckdb_non_fs_and_disabled_persistence_untouched(tmp_path):
    data = {
        "sources": [
            {
                "type": "gdrive",
                "object_id": "drive-folder-id",
                "service_user_credentials_file": "creds.json",
                "file_name_pattern": "*.pdf",
            }
        ],
        "vector_db": {"type": "qdrant", "collection": "c"},
        "persistence": {"enabled": False, "path": "state"},
    }
    cfg = load_config(_write(tmp_path, data))
    assert cfg.vector_db.type == "qdrant"
    assert cfg.sources[0].type == "gdrive"
    # Relative paths on non-fs sources are NOT rebased.
    assert cfg.sources[0].service_user_credentials_file == "creds.json"
    assert cfg.persistence.path == "state"


def test_load_config_dict_keeps_relative_verbatim():
    cfg = load_config_dict(_copy(CONFIG))
    assert cfg.sources[0].path == "docs"
    assert cfg.vector_db.path == "store.duckdb"
    assert cfg.persistence.path == "state"
    assert cfg.indexer.ingestion_log_dir == "log"


def test_parser_rule_unless_env_retains_rule_on_mismatch(monkeypatch):
    monkeypatch.delenv("ENABLE_PAID_PARSER", raising=False)

    cfg = load_config_dict(
        {
            "parser": [
                {
                    "match": ["*.video"],
                    "type": "skip",
                    "unless_env": {"ENABLE_PAID_PARSER": "1"},
                }
            ]
        }
    )

    assert cfg.parser is not None
    assert [rule.type for rule in cfg.parser] == ["skip"]


def test_parser_rule_unless_env_omits_rule_on_exact_trimmed_match(monkeypatch):
    monkeypatch.setenv("ENABLE_PAID_PARSER", " 1 ")

    cfg = load_config_dict(
        {
            "parser": [
                {
                    "match": ["*.video"],
                    "type": "skip",
                    "unless_env": {"ENABLE_PAID_PARSER": "1"},
                },
                {"match": ["*.txt"], "type": "utf8"},
            ]
        }
    )

    assert cfg.parser is not None
    assert [rule.type for rule in cfg.parser] == ["utf8"]
