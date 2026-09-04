"""Configuration fingerprint guarding deterministic re-execution.

Chunking-related UDFs run with ``deterministic=True`` (see ``graph.py``), so
on a retraction Pathway *re-runs* them instead of replaying stored results.
That is only sound while they keep producing byte-identical outputs — which a
config edit or a library upgrade between restarts can silently break: a
document indexed under the old settings would be re-cut differently at
deletion time, its old chunk ids would not match, and the vector DB would
keep orphaned rows.

To catch that, the indexer stores the full risk-relevant objects (not just a
hash) in a JSON file **beside** the Pathway persistence directory. Keeping the
file outside Pathway's directory is required: the engine interprets every root
entry as its own persistence key and logs an error for arbitrary files. The
fingerprint contains:

- the splitter config,
- the complete embedding-space identity (never credentials or runtime tuning),
- parser routing,
- versions of the libraries whose behavior shapes the outputs.

On startup with persistence enabled the stored objects are compared with the
current ones. Any difference is reported as an explicit diff with a
human-readable explanation of the risk, and the indexer refuses to start
unless the user confirms — interactively on a TTY, or via
``SERVIETTE_ACCEPT_FINGERPRINT_CHANGES=1`` in non-interactive deployments.
Confirmation updates the stored fingerprint. Legacy files inside a persistence
directory are moved to the sibling location before Pathway starts.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from serviette.config.schema import ServietteConfig

logger = logging.getLogger(__name__)

_FILENAME = "serviette-fingerprint.json"


def _fingerprint_path(directory: Path) -> Path:
    return directory.parent / f"{directory.name}.{_FILENAME}"


_ACCEPT_ENV = "SERVIETTE_ACCEPT_FINGERPRINT_CHANGES"

_RISKS = {
    "splitter": (
        "Documents indexed before this change were cut with the OLD settings. "
        "When such a document is later deleted or modified, its chunks will be "
        "re-computed with the NEW settings, the chunk ids will not match, and "
        "the old vectors will stay in the vector DB as orphans. Safe options: "
        "revert the change, or re-index from scratch (drop the vector "
        "collection and the persistence directory)."
    ),
    "embedder": (
        "Vectors produced by different embedder models are not comparable: "
        "queries embedded with the new model will rank previously indexed "
        "documents nonsensically. Re-indexing from scratch is strongly "
        "recommended."
    ),
    "parser": (
        "Documents indexed before this change were parsed with the OLD "
        "routing (e.g. a different PDF parser or video prompt), so their "
        "text — and therefore their chunks and vectors — may differ from "
        "what the NEW routing would produce. Edits/deletions of such "
        "documents can leave orphaned vectors. If the affected formats "
        "matter, re-index from scratch; note that defaults also resolve "
        "from the environment (installed parser packages, API keys "
        "present), so this can change without touching the config."
    ),
    "libraries": (
        "A library upgrade can change tokenization and therefore chunk "
        "boundaries — with the same orphaned-vectors risk as a splitter "
        "config change. If unsure, re-index from scratch."
    ),
}


def _library_version(module: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(module)
    except Exception:  # noqa: BLE001 - absent/odd packaging must not break startup
        return None


def build_fingerprint(config: ServietteConfig) -> dict[str, Any]:
    embedder = config.embedder.model_dump() if config.embedder else {}
    runtime_embedder_keys = {
        "api_key",
        "device",
        "batch_size",
        "capacity",
        "retries",
        "timeout",
    }
    embedder_identity = {
        key: value for key, value in embedder.items() if key not in runtime_embedder_keys
    }
    from serviette.indexer.graph import ParserRegistry

    return {
        "splitter": config.splitter.model_dump(),
        "parser_plugins": list(config.parser_plugins),
        "parser": ParserRegistry(config.parser).resolved_rules(),
        # Complete vector-space identity, never credentials or execution
        # tuning. Revision, dimensions, prefixes, and model kwargs can change
        # stored vectors and therefore must force an explicit re-index choice.
        "embedder": embedder_identity,
        "libraries": {
            name: _library_version(name)
            for name in ("pathway", "tiktoken", "langchain_text_splitters")
        },
    }


def _diff(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section, risk in _RISKS.items():
        old, new = stored.get(section), current.get(section)
        if old == new:
            continue
        lines.append(f"[{section}] stored: {json.dumps(old, sort_keys=True)}")
        lines.append(f"[{section}] current: {json.dumps(new, sort_keys=True)}")
        lines.append(f"[{section}] risk: {risk}")
    return lines


def _confirm(diff_lines: list[str]) -> bool:
    if os.environ.get(_ACCEPT_ENV) == "1":
        logger.warning("Fingerprint changes accepted via %s=1; proceeding.", _ACCEPT_ENV)
        return True
    if sys.stdin.isatty():
        answer = input(
            "The indexing configuration changed since the last run (see above). "
            "Proceed anyway, accepting these risks? Type 'yes' to continue: "
        )
        return answer.strip().lower() == "yes"
    return False


def check_fingerprint(config: ServietteConfig) -> None:
    """Verify (and maintain) the persisted fingerprint; may abort startup."""

    if not config.persistence.enabled:
        return  # no persisted state to be inconsistent with
    directory = Path(config.persistence.path)
    directory.mkdir(parents=True, exist_ok=True)
    path = _fingerprint_path(directory)
    legacy_path = directory / _FILENAME
    if not path.exists() and legacy_path.exists():
        legacy_path.replace(path)

    current = build_fingerprint(config)
    if not path.exists():
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return

    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable fingerprint file %s; rewriting it.", path)
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return

    diff_lines = _diff(stored, current)
    if not diff_lines:
        return

    logger.critical(
        "The indexing configuration differs from the one this persistence "
        "directory (%s) was built with:\n%s",
        directory,
        "\n".join("  " + line for line in diff_lines),
    )
    if not _confirm(diff_lines):
        raise SystemExit(
            "Refusing to start: the configuration change above can corrupt "
            "incremental updates of already-indexed documents. Either revert "
            "the change, re-index from scratch (drop the vector collection "
            f"and {directory}), or set {_ACCEPT_ENV}=1 / confirm interactively "
            "to accept the risks."
        )
    path.write_text(json.dumps(current, indent=2, sort_keys=True))
    logger.warning("Fingerprint updated to the current configuration.")
