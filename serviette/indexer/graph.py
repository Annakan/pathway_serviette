"""Pathway graph construction for the indexer.

Mirrors the llm-app ``document_indexing`` template as closely as the
"external vector DB" goal allows, and reuses ``pathway.xpacks.llm`` parsers,
splitters and embedders without modification.

Pipeline
--------
``pw.io.fs.read(format="only_metadata")``  ->  parse UDF (reads bytes from the
path, extracts text via an xpack parser, memoised + persistently cached)  ->
splitter (xpack)  ->  flatten to one row per chunk  ->  embedder (xpack)  ->
vector-DB sink (duckdb / pgvector / milvus / qdrant / chroma / weaviate /
pinecone / mongodb) — every sink is a native ``pw.io`` connector writing in
snapshot/upsert mode, so retractions become real deletes in the target store.

Deletion semantics
------------------
The parse UDF is registered ``deterministic=False`` (Pathway's default), so on a
file removal Pathway re-emits the memoised parsed text *negated* and never
re-reads the now-deleted bytes; the retraction flows through split/embed and the
sink removes exactly the matching vectors.

Parse caching
-------------
Cross-restart parse caching is delegated entirely to Pathway:
``pw.udfs.DefaultCache`` stores results on disk (diskcache, LRU-bounded) under
``<persistence dir>/runtime_calls`` whenever persistence is enabled — which is
the default. No caching machinery lives in serviette; disabling persistence also
disables the parse cache (every restart re-fetches and re-parses).
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import inspect
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pathway as pw

from serviette.config.schema import ServietteConfig
from serviette.indexer.sources import Fetcher, make_fetcher, read_source
from serviette.indexer import parsers as _parser_plugins

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for working with Pathway values imperatively
# ---------------------------------------------------------------------------


def _json_to_dict(value: Any) -> dict[str, Any]:
    """Coerce a Pathway ``Json`` (or already-plain) value into a dict."""

    if isinstance(value, dict):
        return value
    # pw.Json exposes the wrapped python value via ``.value`` / ``as_dict``.
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if hasattr(value, "value"):
        return dict(value.value)
    return dict(value)


# ---------------------------------------------------------------------------
# Parser dispatch (xpack parsers, called imperatively because only_metadata
# means we hold paths, not bytes, in the graph)
# ---------------------------------------------------------------------------


class _IngestionReportWriter:
    """On-disk mirror of the ingestion report (M2 D-M2-3/§Error recording).

    One directory per indexer run — ``<log_dir>/ingestion_<YYYYMMDD-HHMMSS>/``
    — holding ``events.jsonl`` (one record per line, flushed per append so a
    crash keeps records) and ``summary.txt`` (human-readable counts by class
    + per-file reasons, rewritten on each record; volume is bounded by the
    number of bad files). In streaming mode both keep appending for the
    process's lifetime.
    """

    def __init__(self, log_dir: Path) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_dir = log_dir / f"ingestion_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._summary_path = self.run_dir / "summary.txt"
        self._records: list[dict[str, Any]] = []

    def record(self, rec: dict[str, Any]) -> None:
        self._records.append(rec)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._rewrite_summary()

    def _rewrite_summary(self) -> None:
        shown = [r for r in self._records if not r.get("deliberate")]
        omitted = len(self._records) - len(shown)
        counts: dict[str, int] = {}
        for rec in shown:
            key = f"{rec['stage']}/{rec['action']}"
            counts[key] = counts.get(key, 0) + 1
        lines = [
            f"Ingestion report — {len(shown)} record(s)"
            + (f" (+{omitted} deliberate route omissions)" if omitted else ""),
            "",
            "Counts by class:",
            *(f"  {k}: {v}" for k, v in sorted(counts.items())),
            "",
            "Records:",
            *(
                f"  [{r['ts']}] {r['stage']}/{r['action']} {r['file']}"
                + (f" ({r['chunk_locator']})" if r.get("chunk_locator") else "")
                + f" — {r['reason']}"
                for r in shown
            ),
        ]
        self._summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ParserRegistry:
    """Rule-based, extension-dispatched xpack parsers.

    We never implement our own parsing: each file is handed to the matching
    ``pathway.xpacks.llm.parsers`` class. User rules (``parser:`` config) are
    checked first; built-in defaults cover the rest with a keyless-first
    policy — every format is enabled out of the box, routed to the best
    parser that needs no API key, and a modality whose only parser requires
    an absent key (audio -> OpenAI Whisper, video -> TwelveLabs) is skipped
    with a warning instead of failing the pipeline.

    Parsers expose their logic via ``__wrapped__(contents)`` returning
    ``list[(text, metadata)]``; some are async, which we drive to completion
    here so the enclosing UDF stays sync.
    """

    _SUFFIXES: ClassVar[dict[str, set[str]]] = {
        "text": {".txt", ".md", ".markdown", ".text", ""},
        "pdf": {".pdf"},
        "image": {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"},
        "audio": {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"},
        "video": {".mp4", ".webm", ".mov", ".mkv", ".avi"},
        # everything else -> office/unstructured
    }

    def __init__(self, rules: list | None = None) -> None:
        self._rules = list(rules or [])
        self._instances: dict[Any, Any] = {}
        self._defaults: dict[str, tuple[str, dict]] = {}
        # Ingestion report: every file that never reaches the vector store
        # and every dropped chunk (route-skip, fetch error, parse failure,
        # normalization failure, chunk drops, zero-chunk red flags) — the
        # in-memory record list behind ``ingestion_report()`` (M2). Every
        # record also logs a per-file WARNING and is mirrored to the
        # on-disk report when a writer is attached (build_graph).
        self._ingestion_records: list[dict[str, Any]] = []
        self.report_writer: _IngestionReportWriter | None = None

    # -- keyless-first defaults ------------------------------------------------

    @staticmethod
    def _importable(module: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(module) is not None

    def _default_for(self, modality: str) -> tuple[str, dict]:
        """Resolve the default (parser type, options) for a modality.

        Preference order per modality: the best parser that runs without an
        API key; a key-requiring parser only when its key is already present
        in the environment; otherwise ``skip``.
        """
        if modality not in self._defaults:
            kind: tuple[str, dict]
            if modality == "text":
                kind = ("utf8", {})
            elif modality == "pdf":
                # Best keyless first: docling (layout + tables) over pypdf (a
                # core dependency, so PDFs parse in every install); the guard
                # keeps even a broken environment skipping, never crashing.
                if self._importable("docling"):
                    kind = ("docling", {})
                elif self._importable("pypdf"):
                    kind = ("pypdf", {})
                else:
                    kind = ("skip", {"reason": "PDF: install serviette[docling] (or pypdf)"})
            elif modality == "image":
                # PaddleOCR is local/keyless; vision parsers need an API key
                # and are opt-in via explicit rules.
                if self._importable("paddleocr"):
                    kind = ("paddle_ocr", {})
                else:
                    kind = ("skip", {"reason": "images: install serviette[ocr] or configure vision_image"})
            elif modality == "audio":
                if os.environ.get("OPENAI_API_KEY"):
                    kind = ("whisper", {})
                else:
                    kind = ("skip", {"reason": "audio: OPENAI_API_KEY not set (Whisper API)"})
            elif modality == "video":
                if os.environ.get("TWELVELABS_API_KEY"):
                    kind = ("twelvelabs_video", {})
                else:
                    kind = ("skip", {"reason": "video: TWELVELABS_API_KEY not set"})
            elif self._importable("unstructured"):
                kind = ("unstructured", {})
            else:
                kind = ("skip", {"reason": "office formats: install serviette[docling]"})
            self._defaults[modality] = kind
        return self._defaults[modality]

    # Modules each parser kind needs at construction time, with the install
    # hint for the error message. Used to fail fast on explicit ``parser:``
    # rules — a config that names an uninstallable parser should stop the
    # indexer at startup, not crash the pipeline on the first matching file.
    _KIND_DEPS: ClassVar[dict[str, tuple[str, str]]] = {
        "docling": ("docling", 'pip install "serviette[docling]"'),
        "pypdf": ("pypdf", "pip install pypdf"),
        "unstructured": ("unstructured", 'pip install "serviette[docling]"'),
        "paddle_ocr": ("paddleocr", 'pip install "serviette[ocr]"'),
    }

    # Every built-in parser kind (the classes dict in ``_get``). Used by
    # ``check_rule_deps`` to reject unknown names — including registered
    # extension parsers — at startup. Kept as a static set so validation
    # does not import pathway.
    _BUILTIN_KINDS: ClassVar[frozenset[str]] = frozenset({
        "utf8", "pypdf", "docling", "unstructured", "paddle_ocr",
        "vision_image", "vision_slide", "whisper", "twelvelabs_video",
    })

    def check_rule_deps(self) -> None:
        """Validate that every explicit rule's parser can actually be built."""

        for rule in self._rules:
            dep = self._KIND_DEPS.get(rule.type)
            if dep and not self._importable(dep[0]):
                module, hint = dep
                raise ValueError(
                    f"parser rule {rule.match} -> {rule.type!r} needs the "
                    f"{module!r} package, which is not installed — {hint}"
                )
            # Reject names that are neither built-in, registered, nor skip —
            # at startup, before the pipeline meets the first matching file.
            if rule.type == "skip" or rule.type in self._BUILTIN_KINDS:
                continue
            if rule.type in _parser_plugins.registered_parsers():
                continue
            known = sorted({
                *self._BUILTIN_KINDS,
                *_parser_plugins.registered_parsers(),
                "skip",
            })
            raise ValueError(
                f"parser rule {rule.match} -> {rule.type!r} is unknown. "
                f"Known types: {', '.join(known)}"
            )

    def resolved_rules(self) -> list[dict]:
        """Full routing picture (user rules + resolved defaults) for the
        persistence fingerprint and logs. Credentials are never included."""

        resolved = [
            {
                "match": r.match,
                "type": r.type,
                "options": {k: v for k, v in r.options.items() if "key" not in k.lower()},
            }
            for r in self._rules
        ]
        for modality in ["text", "pdf", "image", "audio", "video", "office"]:
            kind, options = self._default_for(modality)
            resolved.append({"match": [f"<default:{modality}>"], "type": kind, "options": options})
        return resolved

    # -- dispatch ---------------------------------------------------------------

    def _modality_for(self, suffix: str) -> str:
        suffix = suffix.lower()
        for modality, suffixes in self._SUFFIXES.items():
            if suffix in suffixes:
                return modality
        return "office"

    def route(self, suffix: str, name: str, path: str = "") -> tuple[str, dict, bool]:
        """Public routing query (read-only tooling, e.g. the rAIvisor ``kb
        orphans`` command, resolves expected-skips with exactly the
        indexer's semantics)."""

        return self._route(suffix, name, path)

    def _route(self, suffix: str, name: str, path: str = "") -> tuple[str, dict, bool]:
        """Resolve (kind, options, matched_rule). ``matched_rule`` is True
        when a config rule (not the environment default) resolved the file —
        the deliberate/accidental distinction for route skips (M2 report)."""

        # Match against both the basename and the full connector path so
        # path-aware rules (*.vidprep/bundle.yaml) work alongside legacy
        # basename globs (*.mp4). fnmatch's * crosses "/", so both shapes
        # keep matching. ``name or path`` so path-only connectors still route.
        candidates = [c for c in (name, path) if c] or [f"x{suffix}"]
        for rule in self._rules:
            if any(
                fnmatch.fnmatch(c, pat) for c in candidates for pat in rule.match
            ):
                return rule.type, dict(rule.options), True
        return *self._default_for(self._modality_for(suffix)), False

    def _get(self, kind: str, options: dict):
        key = (kind, tuple(sorted(options.items())))
        if key not in self._instances:
            self._instances[key] = self._resolve_class(kind)(**options)
        return self._instances[key]

    @staticmethod
    def _resolve_class(kind: str) -> type:
        """Built-in xpack classes first, then registered extension parsers."""
        from pathway.xpacks.llm import parsers

        builtin = {
            "utf8": parsers.Utf8Parser,
            "pypdf": parsers.PypdfParser,
            "docling": parsers.DoclingParser,
            "unstructured": parsers.UnstructuredParser,
            "paddle_ocr": parsers.PaddleOCRParser,
            "vision_image": parsers.ImageParser,
            "vision_slide": parsers.SlideParser,
            "whisper": parsers.AudioParser,
            "twelvelabs_video": parsers.TwelveLabsVideoParser,
        }
        if kind in builtin:
            return builtin[kind]
        registered = _parser_plugins.registered_parsers()
        if kind in registered:
            return registered[kind]
        known = sorted({*builtin, *registered, "skip"})
        raise KeyError(
            f"Unknown parser type {kind!r}. Known: {', '.join(known)}"
        )

    def parse(
        self, contents: bytes, suffix: str, name: str = "", path: str = ""
    ) -> list[tuple[str, dict]]:
        """Parse ``contents`` into ``[(text, meta), ...]`` preserving
        per-element metadata (M2). Skip/failure → ``[]`` (was ``""``)."""

        kind, options, matched_rule = self._route(suffix, name, path)
        if kind == "skip":
            reason = options.get("reason", "no parser configured")
            self._record_event(
                file=name or path or suffix, path=path,
                stage="route", action="skip_file",
                reason=f"skip: {reason}",
                # A config skip RULE is a deliberate exclusion (video
                # upstream, bundle internal, corpus exclusions) — the
                # summary suppresses it. Skips resolved from the
                # environment default (keyless-first fallback) stay
                # visible: they signal a gap, not intent.
                deliberate=matched_rule,
            )
            return []
        options.pop("reason", None)
        try:
            parser = self._get(kind, options)
        except (ImportError, KeyError) as exc:
            # Routing guards make this unreachable for the defaults, and
            # check_rule_deps() for explicit rules — this net catches lazy
            # imports inside the xpack parsers themselves. One file must
            # never kill the pipeline.
            self._record_event(
                file=name or path or suffix, path=path,
                stage="route", action="skip_file",
                reason=f"parser {kind!r} unavailable: {exc}", parser=kind,
            )
            return []
        records_before = len(self._ingestion_records)

        def record_from_parser(
            stage: str,
            action: str,
            reason: str,
            chunk_locator: str | None = None,
        ) -> None:
            """``context["record"]`` for takes_context parsers (M2): emit
            ingestion-report records from inside element generation (e.g.
            ``group/oversize_excerpt`` from the vidprep transcript
            splitter)."""

            self._record_event(
                file=name or path or suffix, path=path,
                stage=stage, action=action, reason=reason,
                parser=kind, chunk_locator=chunk_locator,
            )

        context = {"path": path, "name": name, "metadata": {},
                   "record": record_from_parser}
        try:
            # Registered parsers may define parse(contents, context) with
            # takes_context=True (vidprep_bundle needs the path to find
            # sibling deliverables); xpack parsers use __wrapped__(contents).
            if getattr(parser, "takes_context", False):
                result = parser.parse(contents, context)
            else:
                result = parser.__wrapped__(contents)
                if inspect.isawaitable(result):
                    # Some xpack parsers return a bare Awaitable rather than a Coroutine.
                    result = asyncio.run(result)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - one bad file must never kill the pipeline
            self._record_event(
                file=name or path or suffix, path=path,
                stage="parse", action="skip_file",
                reason=f"parse failure ({kind!r}): {exc}", parser=kind,
            )
            return []
        elements = self._normalize_elements(result, kind, name, path, suffix)
        if (
            not elements
            and records_before == len(self._ingestion_records)
            and contents.strip()
        ):
            # Zero-chunk red flag (M2 D-M2-4): a successfully fetched,
            # non-empty file produced no chunks and said nothing — that's
            # always worth a record. Files that already recorded a
            # skip/drop don't double-record.
            self._record_event(
                file=name or path or suffix, path=path,
                stage="parse", action="zero_chunks",
                reason="non-empty file produced no chunks", parser=kind,
            )
        return elements

    def _normalize_elements(
        self,
        result,
        kind: str,
        name: str,
        path: str,
        suffix: str,
    ) -> list[tuple[str, dict]]:
        """Coerce element texts and apply the element-mapper chain (KB
        normalization + validation).

        Mapper protocol (M2 D-M2-3 — no silent drops): a mapper returns a
        mapped dict (claimed), returns ``None`` (pass to the next mapper),
        raises ``SkipFile`` (file-level problem → one ``skip_file`` record,
        file skipped) or ``DropChunk`` (chunk-level problem → one
        ``drop_chunk`` record, file keeps indexing). With mappers
        registered, an element no mapper claims is dropped WITH a record —
        the KB path must never leak native keys vanilla-style. Without
        mappers, native per-element metadata rides through unchanged."""

        mappers = _parser_plugins.element_mappers()
        source_meta = {"name": name, "path": path}
        elements: list[tuple[str, dict]] = []
        for text, meta in result:
            text = str(text) if text else ""
            if not text:
                self._record_event(
                    file=name or path or suffix, path=path,
                    stage="parse", action="drop_chunk",
                    reason="empty element text", parser=kind,
                )
                continue
            meta = dict(meta) if meta else {}
            if mappers:
                try:
                    mapped: dict | None = None
                    for mapper in mappers:
                        mapped = mapper(text, meta, source_meta, kind)
                        if mapped is not None:
                            break
                except _parser_plugins.SkipFile as exc:
                    self._record_event(
                        file=name or path or suffix, path=path,
                        stage="normalize", action="skip_file",
                        reason=exc.reason, parser=kind,
                    )
                    return []
                except _parser_plugins.DropChunk as exc:
                    self._record_event(
                        file=name or path or suffix, path=path,
                        stage=exc.stage, action="drop_chunk",
                        reason=exc.reason, parser=kind,
                        chunk_locator=exc.chunk_locator,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - defensive: a mapper bug must not kill the pipeline
                    self._record_event(
                        file=name or path or suffix, path=path,
                        stage="normalize", action="skip_file",
                        reason=f"metadata normalization failure ({kind!r}): {exc}",
                        parser=kind,
                    )
                    return []
                if mapped is None:
                    self._record_event(
                        file=name or path or suffix, path=path,
                        stage="normalize", action="drop_chunk",
                        reason=f"no element mapper claimed this element ({kind!r})",
                        parser=kind,
                    )
                    continue
                meta = mapped
            elements.append((text, meta))
        return elements

    def pre_chunked(self, suffix: str, name: str, path: str = "") -> bool:
        """True when the routed parser's elements bypass the splitter."""

        kind, options, _ = self._route(suffix, name, path)
        if kind == "skip":
            return False
        options.pop("reason", None)
        try:
            parser = self._get(kind, options)
        except (ImportError, KeyError):
            return False
        return bool(getattr(parser, "pre_chunked", False))

    def _record_event(
        self,
        *,
        file: str,
        path: str,
        stage: str,
        action: str,
        reason: str,
        parser: str | None = None,
        chunk_locator: str | None = None,
        deliberate: bool = False,
    ) -> None:
        """Append one ingestion-report record, log a per-file WARNING, and
        mirror to the on-disk report when a writer is attached.

        ``stage`` ∈ route/fetch/parse/normalize/validate/group;
        ``action`` ∈ skip_file/drop_chunk/zero_chunks/oversize_excerpt
        (M2 spec §Error recording). Warnings are intentionally NOT deduped:
        every record names its file — volume is bounded by the number of
        bad files, which is exactly what the user must see.
        """

        rec: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": file,
            "path": path,
            "stage": stage,
            "action": action,
            "reason": reason,
        }
        if parser is not None:
            rec["parser"] = parser
        if chunk_locator is not None:
            rec["chunk_locator"] = chunk_locator
        if deliberate:
            rec["deliberate"] = True
        self._ingestion_records.append(rec)
        logger.warning(
            "Ingestion %s/%s %r%s — %s",
            stage, action, file,
            f" ({chunk_locator})" if chunk_locator else "",
            reason,
        )
        if self.report_writer is not None:
            self.report_writer.record(rec)

    def ingestion_report(self) -> list[dict[str, Any]]:
        """Every skip/drop record of this run (M2 report; M3 consumes it)."""

        return list(self._ingestion_records)


# ---------------------------------------------------------------------------
# Embedder / splitter builders (xpack)
# ---------------------------------------------------------------------------


def build_xpack_embedder(cfg) -> pw.UDF:
    """Construct a ``pathway.xpacks.llm.embedders`` UDF from config.

    A ``DefaultCache`` strategy is attached so identical chunks are not
    re-embedded across runs (it uses the persistence layer when enabled).
    """

    if cfg.type == "mock":
        # Test-only deterministic embedder; runs without any provider/credentials.
        from serviette.testing import build_mock_embedder

        return build_mock_embedder()

    from pathway.xpacks.llm import embedders

    cache_strategy = pw.udfs.DefaultCache()
    extra = {
        k: v
        for k, v in cfg.model_dump(
            exclude={"type", "model", "api_key", "query_prefix", "document_prefix"}
        ).items()
        if v is not None
    }
    # ``retries: N`` opts into engine-level exponential backoff — the way to
    # survive provider TPM limits on cheap API tiers, where the SDK's couple
    # of quick retries give up long before the minute window resets.
    retries = extra.pop("retries", None)
    common: dict[str, Any] = {"cache_strategy": cache_strategy, **extra}
    if retries:
        common["retry_strategy"] = pw.udfs.ExponentialBackoffRetryStrategy(
            max_retries=int(retries)
        )
    if cfg.model:
        common["model"] = cfg.model

    if cfg.type == "openai":
        return embedders.OpenAIEmbedder(api_key=cfg.api_key, **common)
    if cfg.type == "litellm":
        return embedders.LiteLLMEmbedder(api_key=cfg.api_key, **common)
    if cfg.type in {"sentence_transformer", "sentencetransformer"}:
        model = common.pop("model", None) or "sentence-transformers/all-MiniLM-L6-v2"
        common.pop("cache_strategy", None)  # local model; caching adds little
        udf = embedders.SentenceTransformerEmbedder(model=model, **common)
        # Local models are deterministic: the engine may re-run them on
        # retraction (cheap CPU) instead of memoizing every vector in RAM.
        # The xpack constructor doesn't expose the flag (upstream PR pending),
        # so it is set on the built UDF; the engine reads it at expression
        # build time. API embedders stay memoized — a re-call costs money and
        # is not guaranteed bit-stable. Model changes across restarts are
        # guarded by the persistence fingerprint.
        udf.deterministic = True
        return udf
    if cfg.type == "gemini":
        return embedders.GeminiEmbedder(api_key=cfg.api_key, **common)
    if cfg.type == "bedrock":
        return embedders.BedrockEmbedder(**common)
    raise ValueError(f"Unsupported embedder type: {cfg.type!r}")


def build_xpack_splitter(cfg):
    """Construct an xpack splitter. ``token_count`` is the default."""

    from pathway.xpacks.llm import splitters

    class TailMergingTokenCountSplitter(splitters.TokenCountSplitter):
        """TokenCountSplitter emits a trailing remainder even when it is far
        below min_tokens; a title-ish tail line then becomes a near-empty
        chunk that outranks real content. Fold such a tail into the previous
        chunk instead (single-chunk texts are left untouched)."""

        def chunk(
            self, text: str, metadata: dict = {}, **kwargs  # noqa: B006 - mirrors the xpack splitter signature
        ) -> list[tuple[str, dict]]:
            chunks = super().chunk(text, metadata, **kwargs)
            if len(chunks) >= 2 and len(chunks[-1][0]) < 100:
                text_prev, meta_prev = chunks[-2]
                chunks[-2] = (text_prev.rstrip() + "\n" + chunks[-1][0], meta_prev)
                chunks.pop()
            return chunks

    if cfg.type in {"token_count", "tokencount"}:
        # TokenCountSplitter is token-based; map chunk_size -> max_tokens.
        return TailMergingTokenCountSplitter(max_tokens=cfg.chunk_size)
    if cfg.type in {"recursive", "recursive_character"}:
        return splitters.RecursiveSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
    if cfg.type in {"null", "none"}:
        return splitters.NullSplitter()
    raise ValueError(f"Unsupported splitter type: {cfg.type!r}")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    config: ServietteConfig,
    *,
    embedder: pw.UDF | None = None,
    splitter=None,
) -> pw.Table:
    """Build the indexing graph and return the final embeddings table.

    ``embedder``/``splitter`` can be injected (tests pass a mock embedder);
    otherwise they are built from ``config``.
    """

    config.for_indexer()

    # Load KB parser plugins (config-driven; idempotent per process) before
    # the registry is built so check_rule_deps sees registered types.
    _parser_plugins.load_plugins(config)

    registry = ParserRegistry(config.parser)
    registry.check_rule_deps()
    registry.report_writer = _IngestionReportWriter(
        Path(config.indexer.ingestion_log_dir)
    )
    splitter = splitter if splitter is not None else build_xpack_splitter(config.splitter)
    embedder = embedder if embedder is not None else build_xpack_embedder(config.embedder)

    # -- parse UDF factory ---------------------------------------------------
    # deterministic=False => memoised; never re-runs on retraction (the source
    # object is gone). DefaultCache persists parsed text across restarts on
    # disk when persistence is enabled (LRU-bounded; size from config). The
    # fetcher makes byte retrieval source-specific (local path vs Drive download).
    cache_strategy = pw.udfs.DefaultCache(
        size_limit=config.indexer.parse_cache_size_gb * 2**30
    )

    def make_parse_udf(fetcher: Fetcher):
        @pw.udf(deterministic=False, cache_strategy=cache_strategy)
        def parse_document(metadata: pw.Json) -> list[tuple[str, dict]]:
            meta = _json_to_dict(metadata)
            try:
                contents, suffix = fetcher.fetch(meta)
            except Exception as exc:  # noqa: BLE001 - object may have vanished / be unreadable
                logger.warning("Could not fetch source object %s: %s", meta, exc)
                registry._record_event(
                    file=str(meta.get("name", "")),
                    path=str(meta.get("path", "")),
                    stage="fetch", action="skip_file",
                    reason=f"fetch failure: {exc}",
                )
                return []
            return registry.parse(
                contents, suffix,
                str(meta.get("name", "")), str(meta.get("path", "")),
            )

        return parse_document

    @pw.udf(deterministic=True)
    def doc_pre_chunked(metadata: pw.Json) -> bool:
        """Route-only check: does this file's parser bypass the splitter?

        Cheaper than parse (no byte fetch); the suffix is derived the same
        way every Fetcher does — ``Path(path or name).suffix``."""
        meta = _json_to_dict(metadata)
        name = str(meta.get("name", ""))
        path = str(meta.get("path", ""))
        suffix = os.path.splitext(path or name)[1]
        return registry.pre_chunked(suffix, name, path)

    # Pure-function whitelist: for these splitter types re-running the UDF is
    # guaranteed to reproduce the original chunks, so the engine need not
    # memoize its outputs (a full extra copy of the corpus in RAM). Guarded
    # against config/library drift by the persistence fingerprint (see
    # fingerprint.py). Unknown/future splitter types fall back to memoization.
    _PURE_SPLITTERS = {"token_count", "tokencount", "recursive", "recursive_character", "null", "none"}
    split_is_pure = config.splitter.type in _PURE_SPLITTERS

    @pw.udf(deterministic=split_is_pure)
    def split_elements(
        elements: list[tuple[str, dict]], pre_chunked: bool
    ) -> list[tuple[str, dict]]:
        """Split each element's text (propagating its metadata to every
        sub-chunk) unless the parser marked the input pre-chunked — then
        elements pass through verbatim (vidprep events are already retrieval
        units). Empty texts are dropped."""
        if not elements:
            return []
        out: list[tuple[str, dict]] = []
        if pre_chunked:
            for text, meta in elements:
                if text:
                    out.append((str(text), _json_to_dict(meta) if meta else {}))
            return out
        for text, meta in elements:
            if not text:
                continue
            meta = _json_to_dict(meta) if meta else {}
            for chunk_text, chunk_meta in splitter.chunk(str(text), meta):
                if chunk_text:
                    out.append((str(chunk_text), dict(chunk_meta)))
        return out

    @pw.udf(deterministic=True)
    def merge_metadata(source: pw.Json, element: pw.Json) -> dict:
        """Merge file-level source metadata with per-chunk element metadata
        (element wins on conflict)."""
        return {**_json_to_dict(source), **_json_to_dict(element)}

    @pw.udf(deterministic=True)
    def make_id(meta_json: str, text: str) -> str:
        # meta_json is the canonical per-CHUNK serialization (source + element
        # metadata merged), so the id is reproducible for (chunk metadata, text).
        # Per-chunk (was per-document): two chunks with identical text but
        # different locators (page, timecode) now get distinct ids.
        digest = hashlib.sha256()
        digest.update(meta_json.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    # -- per-source: read (only_metadata) -> parse ----------------------------
    # Each source gets its own fetcher-bound parse UDF; parsed tables share the
    # (_metadata, elements, pre_chunked) schema and are concatenated before
    # splitting/embedding.
    parsed_tables: list[pw.Table] = []
    for i, src in enumerate(config.sources):
        table = read_source(src, name=f"source_{i}")
        parse_document = make_parse_udf(make_fetcher(src))
        parsed_tables.append(
            table.select(
                _metadata=pw.this._metadata,
                elements=parse_document(pw.this._metadata),
                pre_chunked=doc_pre_chunked(pw.this._metadata),
            )
        )

    parsed = (
        parsed_tables[0]
        if len(parsed_tables) == 1
        else pw.Table.concat_reindex(*parsed_tables)
    )

    # -- split -> flatten -> embed -------------------------------------------
    # Per-chunk metadata (M2): split produces [(text, meta)] pairs; flatten
    # expands them. The stored metadata depends on whether an element mapper
    # is registered:
    # - Mapper active (KB ingestion): store the chunk's element metadata
    #   *verbatim* — the mapper already produced the validated frozen envelope
    #   (additionalProperties: false), and merging the fs connector's source
    #   _metadata (path/owner/size/mtime/…) on top would violate the schema
    #   and store something other than what was validated.
    # - No mapper (vanilla serviette): merge source _metadata with element
    #   metadata so parser-native keys (page_number/pages) ride through.
    # meta_json / chunk_id are derived from whichever metadata is stored, so
    # the id is reproducible for (stored metadata, text) in both paths.
    # Pathway tuple indexing ([0]/[1]) follows the DocumentStore pattern
    # (xpacks/llm/document_store.py).
    chunked = parsed.select(
        _metadata=pw.this._metadata,
        chunk=split_elements(pw.this.elements, pw.this.pre_chunked),
    )
    exploded = chunked.flatten(pw.this.chunk)
    # Asymmetric-retrieval models (e5, bge) want a marker prepended to the
    # embedded text only; chunk_id and the stored text stay prefix-free.
    assert config.embedder is not None  # enforced by for_indexer()
    doc_prefix = config.embedder.document_prefix
    chunk_text = pw.this.chunk[0]
    chunk_meta = pw.this.chunk[1]
    if _parser_plugins.element_mappers():
        # KB path: the envelope is the contract — store it alone.
        stored_meta = chunk_meta
    else:
        stored_meta = merge_metadata(pw.this._metadata, chunk_meta)
    meta_json = _metadata_as_json(stored_meta)
    embed_input = (
        pw.apply_with_type(lambda t, _p=doc_prefix: _p + t, str, chunk_text)
        if doc_prefix
        else chunk_text
    )
    # Pathway reserves the column name "id", so the chunk's primary key lives in
    # "chunk_id"; the sinks map it to each backend's id/primary-key field.
    embedded = exploded.select(
        chunk_id=make_id(meta_json, chunk_text),
        text=chunk_text,
        metadata=stored_meta,
        metadata_json=meta_json,
        embedding=embedder(embed_input),
    )

    _write_sink(embedded, config)
    return embedded


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def _write_sink(table: pw.Table, config: ServietteConfig) -> None:
    """Route the embedded table to the configured backend sink.

    ``table`` carries the metadata twice (native ``metadata`` Json and the
    canonical ``metadata_json`` string); each sink selects exactly the columns
    its backend stores, so nothing is written twice and backends that key rows
    internally (qdrant, mongodb) don't carry a dead ``chunk_id`` field.
    """

    vdb = config.vector_db
    assert vdb is not None  # enforced by for_indexer()
    # Backends storing native JSON metadata, keyed by chunk_id.
    plain = table.select(
        chunk_id=pw.this.chunk_id,
        text=pw.this.text,
        metadata=pw.this.metadata,
        embedding=pw.this.embedding,
    )
    # Backends restricted to scalar record metadata: the JSON string form.
    jsonified = table.select(
        chunk_id=pw.this.chunk_id,
        text=pw.this.text,
        metadata=pw.this.metadata_json,
        embedding=pw.this.embedding,
    )
    if vdb.type == "duckdb":
        _write_duckdb(jsonified, vdb)
    elif vdb.type == "pgvector":
        _write_pgvector(plain, vdb)
    elif vdb.type == "milvus":
        _write_milvus(plain, vdb)
    elif vdb.type == "qdrant":
        # No chunk_id: the sink keys points internally; payload = text + metadata.
        _write_qdrant(plain.without(pw.this.chunk_id), vdb)
    elif vdb.type == "chroma":
        _write_chroma(jsonified, vdb)
    elif vdb.type == "weaviate":
        _write_weaviate(jsonified, vdb)
    elif vdb.type == "pinecone":
        _write_pinecone(jsonified, vdb)
    elif vdb.type == "mongodb":
        # No chunk_id: snapshot mode keys documents by the internal _id.
        _write_mongodb(plain.without(pw.this.chunk_id), vdb)
    else:  # pragma: no cover - guarded by schema
        raise ValueError(f"Unsupported vector_db type: {vdb.type!r}")


@pw.udf(deterministic=True)
def _metadata_as_json(metadata: pw.Json) -> str:
    """Serialize the metadata dict to one canonical JSON string.

    Computed per *chunk* (M2: merged source + element metadata) and used as
    the string form stored by backends whose record metadata must be scalar
    (duckdb/chroma/weaviate/pinecone) and as the metadata part of the chunk
    id hash. sort_keys/ensure_ascii keep it byte-stable.
    """

    return json.dumps(_json_to_dict(metadata), sort_keys=True, ensure_ascii=True)


def _write_pgvector(table: pw.Table, vdb) -> None:
    """Write to Postgres/pgvector in snapshot mode so retractions delete rows.

    The target table must exist with an ``embedding vector(n)`` column (see
    docs/README "Config reference"). ``output_table_type='snapshot'`` keeps the
    table as an exact replica of the current chunk set, issuing real
    INSERT/UPDATE/DELETE keyed by ``id``.
    """

    settings = _libpq_settings(vdb.connection_string)
    pw.io.postgres.write(
        table,
        settings,
        vdb.table,
        output_table_type="snapshot",
        primary_key=[table.chunk_id],
    )


def _write_milvus(table: pw.Table, vdb) -> None:
    pw.io.milvus.write(
        table,
        uri=vdb.resolved_uri(),
        collection_name=vdb.collection,
        primary_key=table.chunk_id,
    )


def _write_duckdb(table: pw.Table, vdb) -> None:
    """Write to an embedded DuckDB file via Pathway's native connector.

    Snapshot mode keyed by ``chunk_id`` keeps the table an exact replica of the
    current chunk set (real upserts/deletes). Embeddings land as native
    ``DOUBLE[]`` lists that the server queries in-database with
    ``list_cosine_similarity``. The metadata travels as a JSON string: DuckDB
    compares identifiers case-insensitively and stores ``pw.Json`` as text
    anyway, and one explicit column keeps the accessor symmetric with the
    other backends.
    """

    out = table
    kwargs: dict[str, Any] = {}
    # Release the single-writer file lock between minibatches so a separate
    # server process can answer queries while the streaming indexer runs
    # (pathway builds with detach_between_batches; the accessor retries through the brief lock
    # windows). Older builds keep the previous hold-the-lock behavior.
    if "detach_between_batches" in inspect.signature(pw.io.duckdb.write).parameters:
        kwargs["detach_between_batches"] = True
    else:
        logger.warning(
            "This pathway build lacks duckdb detach_between_batches: while a "
            "streaming indexer runs, a separate server process cannot read "
            "the database file."
        )
    pw.io.duckdb.write(
        out,
        table_name=vdb.table,
        database=vdb.path,
        output_table_type="snapshot",
        primary_key=[out.chunk_id],
        init_mode="create_if_not_exists",
        **kwargs,
    )


def _write_qdrant(table: pw.Table, vdb) -> None:
    """Write points to a pre-created Qdrant collection (schema-driven sink).

    The sink keys points internally per row, so additions/updates/deletions map
    to native upserts/deletes. It binds each of the collection's named vector
    slots to the same-named table column: the ``embedding`` column feeds the
    dense slot ``prepare._prepare_qdrant`` created; ``text`` and ``metadata``
    are not vector slots, so they become the point payload.
    """

    pw.io.qdrant.write(
        table,
        vdb.grpc_url(),
        vdb.collection,
        api_key=vdb.api_key,
    )


def _write_chroma(table: pw.Table, vdb) -> None:
    """Write to a pre-existing ChromaDB collection (cosine ``hnsw:space``)."""

    out = table
    pw.io.chroma.write(
        out,
        vdb.collection,
        primary_key=out.chunk_id,
        embedding=out.embedding,
        document=out.text,
        metadata_columns=[out.metadata],
        host=vdb.host,
        port=vdb.port,
        ssl=vdb.ssl,
        headers=vdb.headers,
        tenant=vdb.tenant,
        database=vdb.database,
    )


def _write_weaviate(table: pw.Table, vdb) -> None:
    """Write to a pre-existing Weaviate collection (object vector + properties)."""

    out = table
    pw.io.weaviate.write(
        out,
        vdb.collection,
        primary_key=out.chunk_id,
        vector=out.embedding,
        http_host=vdb.http_host,
        http_port=vdb.http_port,
        http_secure=vdb.http_secure,
        api_key=vdb.api_key,
    )


def _write_pinecone(table: pw.Table, vdb) -> None:
    """Write to a pre-existing Pinecone index (dimension must match)."""

    out = table
    pw.io.pinecone.write(
        out,
        vdb.index_name,
        primary_key=out.chunk_id,
        vector=out.embedding,
        api_key=vdb.api_key,
        host=vdb.host,
        namespace=vdb.namespace,
        metadata_columns=[out.text, out.metadata],
    )


def _write_mongodb(table: pw.Table, vdb) -> None:
    """Write documents to MongoDB / Atlas in snapshot mode.

    One document per chunk with the embedding as a BSON number array — exactly
    the shape Atlas Vector Search queries with ``$vectorSearch`` (the user
    creates the ``vectorSearch`` index; Pathway cannot know the dimension).
    """

    pw.io.mongodb.write(
        table,
        connection_string=vdb.connection_string,
        database=vdb.database,
        collection=vdb.collection,
        output_table_type="snapshot",
    )


def _libpq_settings(connection_string: str) -> dict[str, Any]:
    """Parse a ``postgresql://`` URL into a pw.io.postgres settings dict."""

    from urllib.parse import unquote, urlparse

    parsed = urlparse(connection_string)
    settings: dict[str, Any] = {}
    if parsed.hostname:
        settings["host"] = parsed.hostname
    if parsed.port:
        settings["port"] = parsed.port
    if parsed.username:
        settings["user"] = unquote(parsed.username)
    if parsed.password:
        settings["password"] = unquote(parsed.password)
    dbname = parsed.path.lstrip("/")
    if dbname:
        settings["dbname"] = dbname
    return settings


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def persistence_config(config: ServietteConfig):
    """Build a ``pw.persistence.Config`` (or None) from config.

    Enabling persistence prevents re-embedding unchanged documents on restart:
    Pathway replays operator state and the embedder's ``DefaultCache`` is backed
    by this layer.
    """

    if not config.persistence.enabled:
        return None
    backend = pw.persistence.Backend.filesystem(config.persistence.path)
    return pw.persistence.Config(backend)


def run_indexer(
    config: ServietteConfig, *, prepare: bool = True, **build_kwargs: Any
) -> None:
    """Prepare the backend, build the graph and run it (streaming).

    ``prepare=False`` is used by spawned worker processes: the spawn parent
    has already created the target (see ``serviette.indexer.main``).
    """

    if config.pathway_license_key:
        pw.set_license_key(config.pathway_license_key)
    config.for_indexer()
    if prepare:
        # Refuse to run against persisted state built with an incompatible
        # config (chunking drift corrupts retractions — see fingerprint.py).
        from serviette.indexer.fingerprint import check_fingerprint

        check_fingerprint(config)
        # Create the target table/collection/index if missing — the connectors
        # auto-create only for duckdb and qdrant (see prepare.py).
        from serviette.indexer.prepare import prepare_backend

        prepare_backend(config)
    build_graph(config, **build_kwargs)
    run_kwargs: dict[str, Any] = {}
    if config.indexer.udf_cache_directory:
        # Spill the non-deterministic UDF memo (parsed texts) to disk
        # instead of RAM (Pathway feature; ignored on older builds with
        # a warning so configs stay portable).
        if "udf_cache_directory" in inspect.signature(pw.run).parameters:
            run_kwargs["udf_cache_directory"] = config.indexer.udf_cache_directory
        else:
            logger.warning(
                "indexer.udf_cache_directory is set but this pathway build "
                "does not support pw.run(udf_cache_directory=...); the UDF "
                "cache stays in memory."
            )
    if config.indexer.monitoring_http_port:
        # The engine's own observability server (Rust, per worker process):
        # GET /status and GET /metrics (Prometheus) on 127.0.0.1:(base + id).
        os.environ["PATHWAY_MONITORING_HTTP_PORT"] = str(
            config.indexer.monitoring_http_port
        )
        run_kwargs["with_http_server"] = True
    # The interactive monitoring dashboard redraws the terminal with escape
    # codes; under `serviette up` both children share one console and it would
    # wipe the server's logs. Metrics live on the HTTP monitoring server
    # (indexer.monitoring_http_port) instead.
    pw.run(
        monitoring_level=pw.MonitoringLevel.NONE,
        persistence_config=persistence_config(config),
        **run_kwargs,
    )
