# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Final condition-graph corpus, catalog lookup, and on-demand retrieval."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.graph.conditions import merge_catalogs
from reweave.evolution.graph.matching import GraphQuery, match_graphs, semantic_query
from reweave.evolution.graph.views import query_semantic

_MODEL_LOCK = threading.Lock()
QWEN_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


@lru_cache(maxsize=1)
def _embedding_model(revision, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        revision=revision,
        device=device,
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=False,
    )


class GraphRetrievalConfig(BaseModel):
    """Host-owned corpus and local Qwen3-Embedding-0.6B execution inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    graphs: dict[str, Path] = Field(
        min_length=1,
        description="One absolute final SemanticGraph JSON path per task name. Historical bundles remain referenced by that graph; do not list historical graph revisions as separate tasks.",
    )
    embedding_revision: str = Field(
        default=QWEN_EMBEDDING_REVISION,
        pattern=r"^[0-9a-f]{40}$",
        description="Pinned official Qwen/Qwen3-Embedding-0.6B weight revision. Used only to bind the vector cache to its encoder, never as an author-visible identity.",
    )
    device: str = Field(
        default="cpu",
        min_length=1,
        description="Explicit SentenceTransformer device, e.g. cpu, mps or cuda:0. Defaults to CPU; never allocates a physical rollout GPU implicitly.",
    )
    cache: Path = Field(
        description="Absolute SQLite path caching exact catalog-definition text vectors for this model revision. Shared across requests and protected by database transactions.",
    )
    solver_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60,
        allow_inf_nan=False,
        description="Per-candidate branch search budget in seconds, checked between expansions; one assignment and certificate may finish after it. Preparation is reported separately. Feasible results retain a bound; timeout is not an exact ranking certificate.",
    )
    max_candidates: int = Field(
        default=100_000,
        ge=1,
        description="Maximum compiled state/relation scope pairs and support/conflict clauses per candidate. Construction refuses larger queries explicitly.",
    )

    @model_validator(mode="after")
    def validate_paths(self):
        if not self.cache.is_absolute() or any(
            not path.is_absolute() for path in self.graphs.values()
        ):
            raise ValueError("retrieval graph and cache paths must be absolute")
        if any(not task.strip() for task in self.graphs):
            raise ValueError("corpus task names must be nonempty")
        if len({path.resolve() for path in self.graphs.values()}) != len(self.graphs):
            raise ValueError("a final graph cannot be registered as multiple tasks")
        return self


class GraphSearch(BaseModel):
    """The same query contract serves whole-graph and local author searches."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    query: GraphQuery | None = Field(
        default=None,
        description="Optional explicit query graph, including a single directed edge. When omitted or null, the Host uses the current round's complete semantic graph afresh; previous local queries never persist as defaults. Before the first semantic graph exists, supply a task hypothesis explicitly.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of candidate task graphs returned.",
    )
    exclude_tasks: tuple[str, ...] = Field(
        default=(),
        description="Exact corpus task names already inspected or outside this search's permitted reference split.",
    )


class TextEmbeddings:
    """One local encoder path with an exact-text, model-bound persistent cache."""

    def __init__(self, config: GraphRetrievalConfig):
        self.config = config

    def encode(self, documents: list[str]) -> tuple[np.ndarray, dict]:
        config = self.config
        config.cache.parent.mkdir(parents=True, exist_ok=True)
        new_texts, tokens = 0, 0
        encoding_seconds = 0.0
        vectors = {}
        with closing(sqlite3.connect(config.cache, timeout=60)) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS encoder (identity TEXT NOT NULL)"
            )
            database.execute(
                "CREATE TABLE IF NOT EXISTS vectors (text TEXT PRIMARY KEY, value BLOB NOT NULL)"
            )
            database.commit()
            # Serialize cache misses as well as writes, avoiding duplicate
            # inference from concurrent authors sharing the same corpus.
            database.execute("BEGIN IMMEDIATE")
            identity = json.dumps(
                [
                    "Qwen/Qwen3-Embedding-0.6B",
                    config.embedding_revision,
                ]
            )
            existing = database.execute("SELECT identity FROM encoder").fetchall()
            if existing and existing != [(identity,)]:
                raise ValueError(
                    "embedding cache belongs to a different model revision"
                )
            if not existing:
                database.execute("INSERT INTO encoder VALUES (?)", (identity,))
            unique = list(dict.fromkeys(documents))
            for text in unique:
                row = database.execute(
                    "SELECT value FROM vectors WHERE text = ?", (text,)
                ).fetchone()
                if row is not None:
                    vectors[text] = np.frombuffer(row[0], dtype="<f4")
            missing = [text for text in unique if text not in vectors]
            if missing:
                started = time.perf_counter()
                with _MODEL_LOCK:
                    model = _embedding_model(config.embedding_revision, config.device)
                    lengths = [
                        len(ids)
                        for ids in model.tokenizer(missing, truncation=False)[
                            "input_ids"
                        ]
                    ]
                    if max(lengths) > min(model.max_seq_length, 8192):
                        raise ValueError(
                            "catalog text exceeds the encoding budget; shorten the lookup explicitly, never truncate definitions"
                        )
                    values = np.asarray(
                        model.encode(
                            missing,
                            batch_size=8,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            prompt="",
                            show_progress_bar=False,
                        ),
                        dtype="<f4",
                    )
                encoding_seconds = time.perf_counter() - started
                if (
                    values.ndim != 2
                    or values.shape[0] != len(missing)
                    or values.shape[1] == 0
                    or not np.isfinite(values).all()
                    or np.any(np.linalg.norm(values, axis=1) == 0)
                ):
                    raise ValueError("local encoder returned invalid vectors")
                if vectors and len(next(iter(vectors.values()))) != values.shape[1]:
                    raise ValueError(
                        "embedding dimension changed within the same cache"
                    )
                for text, vector in zip(missing, values, strict=True):
                    vectors[text] = vector
                    database.execute(
                        "INSERT INTO vectors VALUES (?, ?)", (text, vector.tobytes())
                    )
                new_texts = len(missing)
                tokens = sum(lengths)
            database.commit()
        return np.asarray([vectors[text] for text in documents], dtype=float), {
            "new_texts": new_texts,
            "input_tokens": tokens,
            "encoding_seconds": encoding_seconds,
            "model": "Qwen/Qwen3-Embedding-0.6B",
        }


class GraphRetrieval:
    """A read-only corpus snapshot; originals are loaded only by explicit reads."""

    def __init__(self, config: GraphRetrievalConfig, *, admitted_graphs=None):
        self.config = config
        self.encoder = TextEmbeddings(config)
        self.graphs = {}
        self.queries = {}
        self.paths = {}
        seen = []
        if admitted_graphs is not None and set(admitted_graphs) != set(config.graphs):
            raise ValueError("admitted graph partition differs from retrieval config")
        for task, path in sorted(config.graphs.items()):
            graph = (
                admitted_graphs[task]
                if admitted_graphs is not None
                else SemanticGraph.model_validate_json(path.read_text())
            )
            if admitted_graphs is None and graph in seen:
                raise ValueError("the same final graph was registered more than once")
            seen.append(graph)
            self.graphs[task] = graph
            self.queries[task], self.paths[task] = semantic_query(
                graph, witness_paths=True
            )

        self.catalog = merge_catalogs(
            *(query.catalog for query in self.queries.values())
        )

    @classmethod
    def from_file(cls, path: Path):
        return cls(GraphRetrievalConfig.model_validate_json(path.read_text()))

    def catalog_documents(self):
        return [
            json.dumps(definition.model_dump(), ensure_ascii=False, sort_keys=True)
            for definition in self.catalog.values()
        ]

    def index(self) -> dict:
        documents = self.catalog_documents()
        _, usage = self.encoder.encode(documents)
        return {
            "tasks": len(self.graphs),
            "fluents": len(self.catalog),
            "embedding": usage,
        }

    def read_catalog(self, text: str | None = None, *, top_k=12) -> dict:
        """Qwen suggests vocabulary coordinates; it never judges state values."""
        if not 1 <= top_k <= 50:
            raise ValueError("catalog top_k must be between 1 and 50")
        keys = list(self.catalog)
        usage = None
        if text is not None:
            if not text.strip():
                raise ValueError("catalog lookup text must be nonempty")
            vectors, usage = self.encoder.encode([text, *self.catalog_documents()])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            order = np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")[:top_k]
            keys = [keys[index] for index in order]
        return {
            "catalog": {key: self.catalog[key].model_dump(mode="json") for key in keys},
            "total_fluents": len(self.catalog),
            "embedding": usage,
            "meaning": "Reuse definitions and value domains verbatim. Lookup similarity suggests coordinates only; it does not establish physical equivalence, opposite values or implication.",
        }

    def search(
        self, request: GraphSearch, *, default_query: GraphQuery | None = None
    ) -> dict:
        query = request.query if request.query is not None else default_query
        if query is None:
            raise ValueError(
                "no current semantic graph is available; supply an explicit query hypothesis"
            )
        merge_catalogs(self.catalog, query.catalog)
        if set(request.exclude_tasks) - self.graphs.keys():
            raise ValueError(
                "exclude_tasks contains a task outside the configured corpus"
            )
        tasks = [task for task in self.graphs if task not in request.exclude_tasks]
        if not tasks:
            return {
                "matches": [],
                "reason": "All configured corpus tasks were explicitly excluded.",
            }
        started = time.perf_counter()
        results = []
        for task in tasks:
            result = match_graphs(
                query,
                self.queries[task],
                seconds=self.config.solver_seconds,
                max_candidates=self.config.max_candidates,
            )
            connections = []
            for relation in result["relations"]:
                if relation["query"][0] != "transition":
                    continue
                reference_id = relation["reference"][1]
                chain = self.paths[task][reference_id]
                connections.append(
                    {
                        "query_interface": relation["query"][1],
                        "reference_interface": reference_id,
                        "coverage": relation["coverage"],
                        "witness_path": [
                            {"edge_id": edge, "witness_index": index}
                            for edge, index in chain
                        ],
                    }
                )
            omissions = len(
                set(self.queries[task].uncertainties)
                - set(self.graphs[task].matching.uncertainties)
            )
            results.append(
                {
                    "task": task,
                    **result,
                    "connections": connections,
                    "path_omissions": omissions,
                }
            )
        results.sort(key=lambda item: (item["distance"], item["task"]))
        return {
            "matches": results[: request.top_k],
            "query": {
                "source": "explicit" if request.query is not None else "current_graph",
                "nodes": len(query.nodes),
            },
            "matching_seconds": time.perf_counter() - started,
            "evaluated_tasks": len(tasks),
            "ranking_certified": all(result["optimal"] for result in results),
            "meaning": "Distance is unsupported query coverage under one consistent object/state alignment and fixed catalog. Zero is full support, not equal programs, physical transfer success or causal credit. Zero-coverage structural alignments are reported explicitly. Read witnesses and programs on demand.",
        }

    def read(self, task: str, *, view: str, identity: str | None = None) -> dict:
        if task not in self.graphs:
            raise ValueError("unknown corpus task")
        if view == "conditions":
            matching = self.graphs[task].matching
            if identity is not None:
                raise ValueError(
                    "conditions view selects the graph; use node/transitions for individual evidence"
                )
            return {
                "task": task,
                "matching": matching.model_dump(mode="json"),
                "projection_uncertainties": sorted(
                    set(self.queries[task].uncertainties) - set(matching.uncertainties)
                ),
            }
        if view not in {"summary", "node", "transitions", "program"}:
            raise ValueError("unsupported reference graph view")
        return {
            "task": task,
            **query_semantic(
                self.graphs[task], view=view, identity=identity, graph_id=task
            ),
        }

    def originals(self) -> dict[str, dict]:
        """Explicit corpus references extend only the Host's read inventory."""
        originals = {}

        for graph in self.graphs.values():
            for reference in graph.original_references():
                path = reference.path
                if not Path(path).is_absolute():
                    raise ValueError(
                        "corpus originals must use accessible absolute paths"
                    )
                value = {
                    "artifact_id": reference.artifact_id,
                    "bytes": reference.bytes,
                }
                if path in originals and originals[path] != value:
                    raise ValueError("corpus originals disagree on file provenance")
                originals[path] = value
        return originals
