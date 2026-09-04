"""Unit tests for the persistence configuration fingerprint."""

from __future__ import annotations

import json

import pytest

from serviette.config.schema import load_config_dict
from serviette.indexer.fingerprint import (
    _FILENAME,
    _fingerprint_path,
    build_fingerprint,
    check_fingerprint,
)


def _config(
    tmp_path,
    chunk_size=512,
    embedder_model=None,
    enabled=True,
    embedder_extra=None,
):
    embedder = {
        "type": "openai",
        "model": embedder_model,
        "api_key": "sk-SECRET",
    }
    embedder.update(embedder_extra or {})
    return load_config_dict(
        {
            "sources": [{"type": "fs", "path": "/data"}],
            "vector_db": {"type": "duckdb", "path": str(tmp_path / "x.duckdb")},
            "embedder": embedder,
            "splitter": {"type": "token_count", "chunk_size": chunk_size},
            "persistence": {"enabled": enabled, "path": str(tmp_path / "persist")},
        }
    )


def test_first_run_writes_fingerprint(tmp_path):
    config = _config(tmp_path)
    check_fingerprint(config)
    stored = json.loads(_fingerprint_path(tmp_path / "persist").read_text())
    assert stored["splitter"]["chunk_size"] == 512


def test_secrets_never_stored(tmp_path):
    fp = build_fingerprint(_config(tmp_path))
    assert "sk-SECRET" not in json.dumps(fp)


def test_embedding_identity_keeps_semantics_and_drops_runtime_tuning(tmp_path):
    fp = build_fingerprint(
        _config(
            tmp_path,
            embedder_model="Qwen/Qwen3-Embedding-0.6B",
            embedder_extra={
                "revision": "abc123",
                "truncate_dim": 1024,
                "query_prefix": "Instruct: retrieve\nQuery: ",
                "document_prefix": "",
                "model_kwargs": {"dtype": "float16"},
                "device": "cuda",
                "batch_size": 1,
                "capacity": 4,
                "retries": 8,
            },
        )
    )["embedder"]

    assert fp["revision"] == "abc123"
    assert fp["truncate_dim"] == 1024
    assert fp["query_prefix"] == "Instruct: retrieve\nQuery: "
    assert fp["model_kwargs"] == {"dtype": "float16"}
    assert not {"api_key", "device", "batch_size", "capacity", "retries"} & fp.keys()


def test_unchanged_config_passes(tmp_path):
    check_fingerprint(_config(tmp_path))
    check_fingerprint(_config(tmp_path))  # no prompt, no exception


def test_changed_splitter_aborts_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, chunk_size=512))
    with pytest.raises(SystemExit, match="Refusing to start"):
        check_fingerprint(_config(tmp_path, chunk_size=256))


def test_env_confirmation_accepts_and_updates(tmp_path, monkeypatch):
    check_fingerprint(_config(tmp_path, chunk_size=512))
    monkeypatch.setenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", "1")
    check_fingerprint(_config(tmp_path, chunk_size=256))
    stored = json.loads(_fingerprint_path(tmp_path / "persist").read_text())
    assert stored["splitter"]["chunk_size"] == 256
    # Accepted once — the updated fingerprint now matches without the env.
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES")
    check_fingerprint(_config(tmp_path, chunk_size=256))


def test_legacy_inner_fingerprint_moves_beside_pathway_state(tmp_path):
    config = _config(tmp_path)
    directory = tmp_path / "persist"
    directory.mkdir()
    legacy = directory / _FILENAME
    legacy.write_text(json.dumps(build_fingerprint(config)))

    check_fingerprint(config)

    assert not legacy.exists()
    assert _fingerprint_path(directory).is_file()


def test_embedder_change_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-small"))
    with pytest.raises(SystemExit):
        check_fingerprint(_config(tmp_path, embedder_model="text-embedding-3-large"))


def test_embedder_revision_change_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVIETTE_ACCEPT_FINGERPRINT_CHANGES", raising=False)
    check_fingerprint(_config(tmp_path, embedder_extra={"revision": "first"}))
    with pytest.raises(SystemExit):
        check_fingerprint(_config(tmp_path, embedder_extra={"revision": "second"}))


def test_disabled_persistence_skips_check(tmp_path):
    check_fingerprint(_config(tmp_path, enabled=False))
    assert not _fingerprint_path(tmp_path / "persist").exists()


def test_conditional_parser_rule_changes_effective_fingerprint(monkeypatch):
    def config():
        return load_config_dict(
            {
                "parser": [
                    {
                        "match": ["*.mp4", "*.webm", "*.mov", "*.mkv", "*.avi"],
                        "type": "skip",
                        "unless_env": {"ENABLE_PAID_VIDEO": "1"},
                        "options": {"reason": "paid video disabled"},
                    }
                ],
                "embedder": {"type": "mock"},
            }
        )

    monkeypatch.setenv("TWELVELABS_API_KEY", "provider-key")
    monkeypatch.delenv("ENABLE_PAID_VIDEO", raising=False)
    disabled = build_fingerprint(config())["parser"]
    monkeypatch.setenv("ENABLE_PAID_VIDEO", "1")
    enabled = build_fingerprint(config())["parser"]

    assert disabled != enabled
    assert disabled[0]["type"] == "skip"
    assert any(
        rule["match"] == ["<default:video>"] and rule["type"] == "twelvelabs_video"
        for rule in enabled
    )
