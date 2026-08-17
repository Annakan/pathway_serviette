"""Parser-registration extension point (M2).

Serviette ships a closed set of built-in parsers (the xpack classes in
``ParserRegistry._get``). Real knowledge bases need their own parsers —
``raivisor`` registers a ``vidprep_bundle`` parser here — but serviette must
not import KB code (dependency direction: ``raivisor → serviette``).

This module is the generic seam. Two registration surfaces:

- ``register_parser(name, cls)`` — a parser class is added to the lookup the
  ``ParserRegistry`` consults after its built-ins. Registered parsers may
  define ``parse(contents, context) -> list[tuple[str, dict]]`` (with
  ``context = {"path", "name", "metadata"}``) and a ``pre_chunked = True``
  class attribute; classes without ``parse`` fall back to the xpack
  ``__wrapped__(contents)`` protocol.
- ``register_element_mapper(fn)`` — ``fn(text, meta, source_meta,
  parser_kind) -> dict | None`` applied to every parsed element. ``None``
  drops the element; raising fails the *file* (skipped with a WARNING, never
  reaching the vector store). This is where KB-side envelope normalization +
  schema validation hooks in.

Plugin loading is config-driven (``ServietteConfig.parser_plugins``): the
indexer imports each module and calls its ``register(config)`` if defined.
That survives ``serviette up``'s subprocess model — children re-read the
config and re-import — so no caller-side wiring is required.

The registry is process-global; tests use ``unregister_parser`` /
``clear_element_mappers`` for isolation.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from serviette.config.schema import ServietteConfig

logger = logging.getLogger(__name__)

# A registered parser class. Built-ins are NOT stored here — they live in
# ParserRegistry._get's classes dict; this dict is consulted only as a
# fallback, so a KB name can never shadow a built-in.
_PARSER_CLASSES: dict[str, type] = {}

# Element mappers applied, in registration order, to every parsed element.
# The first mapper to return a dict wins; None passes to the next mapper.
# DropChunk/SkipFile below carry the drop/skip reason into the ingestion
# report — the silent-None drop is banned (M2 D-M2-3). In practice exactly
# one mapper is registered (the KB normalization layer).
_ELEMENT_MAPPERS: list[Callable[[str, dict, dict, str], dict | None]] = []


class SkipFile(Exception):
    """Mapper signal: skip the whole file, record the reason.

    Raised for file-level problems (fetch/route/parse failures are handled
    pipeline-side; mappers raise this for unsupported parser output,
    unknown source kind, path outside corpus roots, missing session dir).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DropChunk(Exception):
    """Mapper signal: drop this element, record the reason; the file keeps
    indexing (M2 D-M2-3 — skipping a whole file for one bad chunk is
    forbidden). ``stage`` defaults to ``validate`` (schema validation is
    the common case); ``chunk_locator`` identifies the dropped chunk
    (page/slide number, timecode window) when known.
    """

    def __init__(
        self,
        reason: str,
        *,
        stage: str = "validate",
        chunk_locator: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage
        self.chunk_locator = chunk_locator

# Modules already imported by load_plugins, so a second build_graph call in
# the same process (tests, re-runs) does not re-invoke register() and stack
# duplicate mappers. register_parser itself is idempotent (same name
# overwrites), but mapper lists are append-only without this guard.
_LOADED_PLUGINS: set[str] = set()


def register_parser(name: str, parser_cls: type) -> None:
    """Register a parser class under ``name`` (overwrites a prior binding)."""
    _PARSER_CLASSES[name] = parser_cls


def unregister_parser(name: str) -> None:
    """Remove a registered parser (test isolation). No-op if absent."""
    _PARSER_CLASSES.pop(name, None)


def registered_parsers() -> dict[str, type]:
    """Snapshot of the registered parser classes (name → class)."""
    return dict(_PARSER_CLASSES)


def register_element_mapper(
    fn: Callable[[str, dict, dict, str], dict | None],
) -> None:
    """Append an element mapper. Deduped by function identity."""
    if fn not in _ELEMENT_MAPPERS:
        _ELEMENT_MAPPERS.append(fn)


def clear_element_mappers() -> None:
    """Remove every element mapper (test isolation)."""
    _ELEMENT_MAPPERS.clear()


def element_mappers() -> list[Callable[[str, dict, dict, str], dict | None]]:
    """Snapshot of the registered element mappers."""
    return list(_ELEMENT_MAPPERS)


def reset_registry() -> None:
    """Clear parsers + mappers + the loaded-plugin set (test isolation)."""
    _PARSER_CLASSES.clear()
    _ELEMENT_MAPPERS.clear()
    _LOADED_PLUGINS.clear()


def load_plugins(config: ServietteConfig) -> None:
    """Import ``config.parser_plugins`` modules and call ``register(config)``.

    Idempotent per process: a module is imported and its ``register`` invoked
    at most once, even across repeated ``build_graph`` calls. Importing the
    module twice is harmless (Python caches it); the guard stops
    ``register`` re-appending mappers.
    """
    for dotted in config.parser_plugins:
        if dotted in _LOADED_PLUGINS:
            continue
        module = importlib.import_module(dotted)
        register = getattr(module, "register", None)
        if callable(register):
            register(config)
        _LOADED_PLUGINS.add(dotted)
