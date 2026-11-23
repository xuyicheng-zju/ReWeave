# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Workspace-bound development tools served by the Host to native authors.

Validation must run outside the author's filesystem restriction: the production
program sandbox creates read-only sources before entering its stricter worker.
This service accepts no command, source root or workspace from its client.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from reweave.evolution.authoring.candidate import host_input
from reweave.evolution.authoring.client import (
    DevelopmentToolError,
    call_development_tool,
    development_client,
)
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.evidence.records import artifact_record

__all__ = [
    "AuthorDevelopmentTools",
    "DevelopmentToolError",
    "call_development_tool",
    "development_client",
]


def read_image_original(original):
    """Read verified original image bytes; return None for textual originals."""
    path = Path(original.path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return None
    from PIL import Image

    from reweave.evolution.contracts import EvidenceFile

    if original.pointer:
        raise ValueError("image originals do not accept JSON pointers")
    EvidenceFile(
        path=original.path, artifact_id=original.artifact_id, bytes=original.bytes
    ).verify()
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        source.verify()
        mime = Image.MIME[source.format]
    return {
        "type": "image",
        "mimeType": mime,
        "data": base64.b64encode(raw).decode(),
    }


async def _check_environment() -> dict:
    from reweave.evolution.bundle import ProgramLimits, PythonFile, PythonProgram
    from reweave.evolution.infrastructure.async_work import run_blocking
    from reweave.runtime.sandbox import ProgramSandbox

    sandbox = ProgramSandbox()
    program = PythonProgram(
        entrypoint="probe:run",
        files=(
            PythonFile(
                path="probe.py",
                source='def run(runtime, context):\n    return runtime.observe("probe", {})\n',
            ),
        ),
    )
    await run_blocking(sandbox.validate_program_import, program, wall_time_s=30)

    async def invoke(operation, name, arguments):
        if (operation, name, arguments) != ("observe", "probe", {}):
            raise ValueError("preflight runtime call differs from the probe")
        return {"probe": "verified"}

    result = await sandbox.run_program(
        program,
        ProgramLimits(runtime_calls=1, physical_calls=0, model_calls=0, wall_time_s=30),
        {},
        invoke,
        input_schema={"type": "object"},
    )
    if result != {"probe": "verified"}:
        raise ValueError("development runtime preflight returned the wrong result")
    return {"environment_ready": True, "physical_execution_tested": False}


def candidate_validation_tool():
    """One validation protocol for ordinary and shared C/W/L authors."""
    from mcp import types

    return types.Tool(
        name="validate_candidate",
        description="Check a candidate while editing: environment checks local execution; code checks files, scope, schemas, imports and synthetic components; interaction additionally replays .scratch/tests/interaction.json through the production runtime. Run interaction after writing or repairing the suite so its errors can be fixed in this same turn. This is a preview; the Host repeats final admission after the native turn.",
        inputSchema={
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["environment", "code", "interaction"],
                    "default": "code",
                },
            },
            "additionalProperties": False,
        },
    )


async def validate_author_candidate(arguments, code_check):
    phase = arguments.get("phase", "code")
    if phase == "environment":
        return await _check_environment()
    if phase == "interaction":
        return await code_check(interactions=True)
    if phase != "code":
        raise ValueError("unknown validation phase")
    return await code_check()


class AuthorDevelopmentTools:
    """One authenticated endpoint, workspace and audit stream per proposal."""

    def __init__(
        self,
        workspace: Path,
        audit: Path,
        evidence_roots: dict[str, Path] | None = None,
        retrieval_config: Path | None = None,
    ):
        from mcp import types
        from mcp.server.lowlevel import Server

        from reweave.planner.utils.http_mcp_server import HttpMcpServer

        self.workspace = workspace.resolve()
        prepared = json.loads((self.workspace / "context/input.json").read_text())
        self.kind = prepared["kind"]
        self.topology_disabled = prepared.get("topology_disabled", False)
        self.references = AuthorReferences.load(self.workspace)
        self.retrieval = None
        if retrieval_config is not None and not self.topology_disabled:
            from reweave.evolution.graph.retrieval import GraphRetrieval

            self.retrieval = GraphRetrieval.from_file(retrieval_config)
            for path, metadata in self.retrieval.originals().items():
                if (
                    path in self.references.originals
                    and self.references.originals[path] != metadata
                ):
                    raise ValueError(
                        "reference corpus conflicts with current original provenance"
                    )
                self.references.originals[path] = metadata
            self.references.save()
        self.private = self.workspace / ".host_inputs"
        if not self.private.is_dir():
            self.private = host_input(self.workspace, "executed_sources.json").parent
        self.executed_sources = json.loads(
            (self.private / "executed_sources.json").read_text()
        )
        self.trace_workspace = self.private / "evidence"
        self.evidence_roots = {
            identity: root.resolve()
            for identity, root in (evidence_roots or {}).items()
        }
        self.audit = audit
        self.reports = audit.parent / "development_reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        app = Server("author", version="1.0.0")

        @app.list_tools()
        async def list_tools():
            catalog = []
            catalog.extend(
                [
                    candidate_validation_tool(),
                ]
            )
            catalog.extend(
                [
                    types.Tool(
                        name="read_trace",
                        description="Start with view=executions for the physical-tool sequence, all indexed Planner feedback and recorded Workflow lifecycle events, including zero-physical-action calls. execution_ref selects one execution and its linked Planner feedback; Workflow events remain episode-wide. start_step/end_step/features are valid only for telemetry; timeline and diagnostics reject them. Payload references resolve exact originals. Query relevant slices instead of rereading the batch.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "view": {
                                    "type": "string",
                                    "enum": [
                                        "executions",
                                        "timeline",
                                        "telemetry",
                                        "diagnostics",
                                        "online_workflows",
                                    ],
                                },
                                "execution_ref": {"type": "string"},
                                "start_step": {"type": "integer", "minimum": 0},
                                "end_step": {"type": "integer", "minimum": 0},
                                "features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["evidence_id", "view"],
                            "additionalProperties": False,
                        },
                    ),
                    types.Tool(
                        name="read_sensor_snapshot",
                        description="Read one admitted historical RGB-D snapshot. Completed local sensor archives are read directly without network or duplicate cache; older server-backed episodes fetch one NPZ and metadata into a read-only cache. Supply a trace evidence handle and its exact observation_ref from read_trace. Returns local array paths, metadata, original references and measured download bytes/latency. No live capture, model inference or actions; connection/authentication remain Host-only.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "observation_ref": {
                                    "type": "string",
                                    "pattern": "^[A-Za-z0-9_-]{1,128}:(?:0|[1-9][0-9]*):[0-9a-f]{32}$",
                                    "description": "Exact sensor snapshot ID from read_trace's sensor_snapshot entry, not a file: handle or JSON pointer. For ordinary local observation files use read_original.",
                                },
                            },
                            "required": ["evidence_id", "observation_ref"],
                            "additionalProperties": False,
                        },
                    ),
                    types.Tool(
                        name="read_video",
                        description="Inspect selected frames of an indexed episode video as native images, with source/artifact/frame/time references. Start with frames=[] for metadata, then request the interval relevant to a concrete physical question. Frame indices are not automatically physical steps. Works for graph analysis and mutation; no physics runs.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "path": {
                                    "type": "string",
                                    "description": "Episode-relative video path from context/trace/<evidence_id>/index.json media.",
                                },
                                "frames": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0},
                                    "uniqueItems": True,
                                    "description": "Sorted explicit video frame indices; empty reads metadata only.",
                                },
                            },
                            "required": ["evidence_id", "path", "frames"],
                            "additionalProperties": False,
                        },
                    ),
                    types.Tool(
                        name="read_original",
                        description="Read the complete immutable original selected by a file reference from this workspace's graph, trace, history or validation report. Pointer suffixes select exact JSON/JSONL values; the Host resolves provenance. Original text, code and payloads are preserved.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "reference": {
                                    "type": "string",
                                }
                            },
                            "required": ["reference"],
                            "additionalProperties": False,
                        },
                    ),
                    types.Tool(
                        name="read_source",
                        description="Read one exact UTF-8 file from a historical executed-source inventory. Use the source_id and relative path listed in context/source_inventory.json. This is targeted provenance access, not repository browsing.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "path": {"type": "string"},
                            },
                            "required": ["source_id", "path"],
                            "additionalProperties": False,
                        },
                    ),
                ]
            )
            if self.kind == "graph_analysis":
                catalog = [
                    tool for tool in catalog if tool.name != "validate_candidate"
                ]
                catalog.append(
                    types.Tool(
                        name="validate_graph",
                        description="Check analysis/result.json with the same schema, reference, full-history coverage and action-continuity validator used at final publication. Returns a compact receipt and full report path. Run on an early complete draft; repair concrete errors. This does not certify semantics or publish the graph; finish the native turn after the required checks pass.",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    )
                )
            catalog.append(
                types.Tool(
                    name="read_graph",
                    description="Read execution evidence, the semantic transition/program graph, or Host map. summary returns a compact index. Semantic program selects its implementation and dependencies; map version/episode/program/node selects actual progress and witnesses. Map transitions selects a literal semantic edge ID; witness selects an issued transition handle. comparison pairs a version with distinct historical receipts on the same condition. Execution episode supports frame/feature selection. Complete originals remain available; select layer explicitly.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "layer": {
                                "type": "string",
                                "enum": ["execution", "semantic", "map"],
                            },
                            "view": {
                                "type": "string",
                                "enum": [
                                    "summary",
                                    "features",
                                    "node",
                                    "episode",
                                    "transitions",
                                    "witness",
                                    "program",
                                    "version",
                                    "comparison",
                                ],
                            },
                            "identity": {"type": "string"},
                            "first_frame": {"type": "integer", "minimum": 0},
                            "last_frame": {"type": "integer", "minimum": 0},
                            "features": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                    },
                )
            )
            if self.retrieval is not None:
                from reweave.evolution.graph.retrieval import GraphSearch

                catalog.extend(
                    [
                        types.Tool(
                            name="retrieve_graphs",
                            description="Retrieve reference task graphs using ONE semantic and directed-structure matching objective. With no query, use this round's complete current semantic graph, refreshed from the shared graph on each call. Optionally override it with selected local states/relations or a task hypothesis. Local overrides never persist to subsequent calls or rounds. Returned matches contain correspondences and witness-path IDs. Choose references yourself and exclude inspected tasks for another batch. No raw programs or trajectories are injected.",
                            inputSchema=GraphSearch.model_json_schema(),
                        ),
                        types.Tool(
                            name="read_reference_graph",
                            description="Read a selected reference task's graph, edge witnesses, or historical program only when useful. With no task, list the configured corpus names and sizes. Program bundle handles are read with read_original. References explain prior behavior and uncertainty, not a certified causal or independently executable plugin.",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string"},
                                    "view": {
                                        "type": "string",
                                        "enum": [
                                            "summary",
                                            "node",
                                            "transitions",
                                            "program",
                                            "conditions",
                                            "catalog",
                                        ],
                                    },
                                    "identity": {"type": "string"},
                                    "text": {
                                        "type": "string",
                                        "description": "Optional vocabulary lookup text for view=catalog only. Omit to read the full shared finite-domain catalog.",
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 50,
                                        "description": "Number of catalog definitions suggested by a text lookup; does not filter graph retrieval candidates.",
                                    },
                                },
                                "additionalProperties": False,
                            },
                        ),
                    ]
                )
            if self.topology_disabled:
                catalog = [
                    tool
                    for tool in catalog
                    if tool.name
                    not in {
                        "read_graph",
                        "retrieve_graphs",
                        "read_reference_graph",
                        "validate_graph",
                    }
                ]
            return catalog

        self.catalog = list_tools

        @app.call_tool()
        async def call_tool(name, arguments):
            async with self._lock:
                started = time.time()
                entry = {"tool": name, "arguments": arguments, "started_at": started}
                case_results = []
                value = None
                error = None
                media = []
                validation = name in {"validate_candidate", "validate_graph"}
                try:
                    value = await self._call(
                        name,
                        self._model_arguments(name, dict(arguments or {})),
                        case_results=case_results,
                    )
                    if name in {"read_video", "read_original"}:
                        media = value.pop("_images", [])
                    entry.update(
                        ok=True,
                        result=value
                        if validation or name in {"read_video", "read_sensor_snapshot"}
                        else {"complete": True},
                    )
                except asyncio.CancelledError:
                    if validation:
                        entry["contract_case_results"] = case_results
                    entry.update(
                        ok=False,
                        error_type="CancelledError",
                        error="development request cancelled; submitted workers drained",
                    )
                    raise
                except Exception as caught:
                    error = caught
                    if validation:
                        entry["contract_case_results"] = case_results
                    entry.update(
                        ok=False,
                        error_type=type(caught).__name__,
                        error=str(caught),
                    )
                finally:
                    entry["elapsed_s"] = time.time() - started
                    self.audit.parent.mkdir(parents=True, exist_ok=True)
                    with self.audit.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                complete = value if error is None else entry
                if not validation:
                    response = (
                        self._model_result(name, complete)
                        if error is None
                        else self.references.encode(complete)
                    )
                else:
                    response = self.references.encode(
                        self._validation_receipt(complete, entry)
                    )
                self.references.save()
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text", text=json.dumps(response, ensure_ascii=False)
                        ),
                        *(types.ImageContent(**item) for item in media),
                    ],
                    structuredContent=response,
                    isError=error is not None,
                )

        # The unguessable route is enforced by HttpMcpServer. A sibling cannot
        # query this workspace by discovering its listening port.
        self.server = HttpMcpServer(app, path="/" + secrets.token_urlsafe(32))

    def _validation_receipt(self, complete: dict, entry: dict) -> dict:
        """Persist a complete report and return only decision-relevant status."""
        raw = json.dumps(
            entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        identity = secrets.token_hex(16)
        path = self.reports / f"{identity}.json"
        temporary = self.reports / f".{identity}.tmp"
        temporary.write_bytes(raw)
        os.replace(temporary, path)

        cases = complete.get("contract_case_results", [])
        receipt = {
            key: complete[key]
            for key in (
                "bundle_id",
                "contract_cases",
                "phase",
                "structurally_valid",
                "plan_checked",
                "physical_execution_tested",
                "interaction_tested",
                "environment_ready",
                "causality_verified",
                "forecast_context",
                "graph_id",
                "episode_count",
                "semantics_verified",
                "tool",
                "ok",
                "error_type",
                "elapsed_s",
            )
            if key in complete
        }
        if "error" in complete:
            error = str(complete["error"]).encode()
            receipt["error_bytes"] = len(error)
        if "contract_case_results" in complete:
            receipt["contract_case_results"] = [
                {
                    key: case[key]
                    for key in (
                        "scope",
                        "workflow_id",
                        "case_index",
                        "validation_status",
                    )
                    if key in case
                }
                | (
                    {"error_bytes": len(str(case["error"]).encode())}
                    if "error" in case
                    else {}
                )
                for case in cases
            ]
        receipt["complete_report"] = self.references.reference(**artifact_record(path))
        receipt["report_bytes"] = len(raw)
        return receipt

    def _model_arguments(self, name: str, arguments: dict) -> dict:
        """Resolve the one model-facing contract before invoking Host behavior."""
        if name == "read_original":
            return {
                "reference": self.references.original(
                    arguments["reference"]
                ).model_dump(mode="json")
            }
        result = dict(arguments)
        if name == "read_sensor_snapshot" and set(result) != {
            "evidence_id",
            "observation_ref",
        }:
            raise ValueError("sensor reads accept only evidence_id and observation_ref")
        if name in {"read_trace", "read_video", "read_sensor_snapshot"}:
            result["evidence_id"] = self.references.resolve(
                arguments["evidence_id"], kind="trace"
            )
        elif name == "read_source":
            result["source_id"] = self.references.resolve(
                arguments["source_id"], kind="source"
            )
        elif name == "read_graph" and arguments.get("identity") is not None:
            layer = arguments.get("layer", "execution")
            view = arguments.get("view", "summary")
            kind = {
                ("execution", "episode"): "episode",
                ("execution", "node"): "partition",
                ("execution", "transitions"): "partition",
                ("semantic", "episode"): "episode",
                ("map", "version"): "version",
                ("map", "comparison"): "version",
                ("map", "episode"): "episode",
                ("map", "witness"): "transition",
            }.get((layer, view))
            if kind is not None:
                result["identity"] = self.references.resolve(
                    arguments["identity"], kind=kind
                )
        return result

    async def model_call(
        self, name: str, arguments: dict, *, case_results: list[dict]
    ) -> dict:
        """Share reference resolution with the read-only inquiry endpoint."""
        value = await self._call(
            name, self._model_arguments(name, arguments), case_results=case_results
        )
        result = self._model_result(name, value)
        self.references.save()
        return result

    def _model_result(self, name: str, value: dict) -> dict:
        if name == "read_original":
            return value
        if name == "read_video":
            value = {
                **value,
                "frames": [
                    {
                        "frame_index": frame["frame_index"],
                        "nominal_video_time_s": frame["nominal_video_time_s"],
                    }
                    for frame in value["frames"]
                ],
            }
        return self.references.encode(value)

    async def _call(
        self, name: str, arguments: dict, *, case_results: list[dict]
    ) -> dict:
        if self.topology_disabled and name in {
            "read_graph",
            "retrieve_graphs",
            "read_reference_graph",
            "validate_graph",
        }:
            raise ValueError("topology is disabled for this request")
        from reweave.evolution.infrastructure.async_work import run_blocking

        if name in {"retrieve_graphs", "read_reference_graph"}:
            if self.retrieval is None:
                raise ValueError("no reference graph corpus is configured")
            if name == "retrieve_graphs":
                from reweave.evolution.contracts import SemanticGraph
                from reweave.evolution.graph.matching import semantic_query
                from reweave.evolution.graph.retrieval import GraphSearch

                request = GraphSearch.model_validate(arguments)
                default_query = None
                if request.query is None:
                    draft = self.workspace / "analysis/result.json"
                    if self.kind == "graph_analysis" and draft.exists():
                        payload = self.references.read_json(draft)["graph"]
                    else:
                        payload = self.references.read_json(
                            self.workspace / "context/graph.json"
                        )
                    if payload is not None:
                        default_query, _ = semantic_query(
                            SemanticGraph.model_validate(payload)
                        )
                result = await run_blocking(
                    self.retrieval.search, request, default_query=default_query
                )
                if self.kind == "graph_analysis":
                    from reweave.utils.persistence import atomic_json

                    path = self.audit.parent / "graph_retrieval.json"
                    searches = json.loads(path.read_text()) if path.exists() else []
                    searches.append(
                        {
                            "request": {
                                **request.model_dump(mode="json"),
                                "query": (request.query or default_query).model_dump(
                                    mode="json"
                                ),
                            },
                            "result": result,
                        }
                    )
                    atomic_json(path, searches)
                return result
            if set(arguments) - {"task", "view", "identity", "text", "top_k"}:
                raise ValueError("unknown reference graph read argument")
            if arguments.get("view") == "catalog":
                if "task" in arguments or "identity" in arguments:
                    raise ValueError(
                        "catalog is shared across reference tasks; omit task and identity"
                    )
                return await run_blocking(
                    self.retrieval.read_catalog,
                    arguments.get("text"),
                    top_k=arguments.get("top_k", 12),
                )
            if "text" in arguments or "top_k" in arguments:
                raise ValueError("text and top_k are only valid for catalog lookup")
            if "task" not in arguments:
                if arguments:
                    raise ValueError("select a corpus task before selecting a view")
                return {
                    "tasks": {
                        task: {"nodes": len(graph.nodes), "edges": len(graph.edges)}
                        for task, graph in self.retrieval.graphs.items()
                    }
                }
            return self.retrieval.read(
                arguments["task"],
                view=arguments.get("view", "summary"),
                identity=arguments.get("identity"),
            )
        from reweave.evolution.evidence.trace import query_trace

        if name == "validate_graph":
            from reweave.evolution.contracts import GraphAnalysisRequest, GraphSnapshot
            from reweave.evolution.graph.projection import (
                analysis_graph_receipts,
                project_map,
            )
            from reweave.evolution.graph.refinement import validate_analysis_output
            from reweave.evolution.graph.selection import (
                forecast_context,
                policy_parent,
            )

            if self.kind != "graph_analysis":
                raise ValueError("mutation author has no graph validation authority")
            if arguments:
                raise ValueError("validate_graph accepts no arguments")
            request = GraphAnalysisRequest.model_validate_json(
                (self.private / "analysis_request.json").read_text()
            )
            result = await run_blocking(
                validate_analysis_output, request, self.workspace / "analysis"
            )
            if not request.needs_forecast:
                return {
                    "structurally_valid": True,
                    "graph_id": "draft:" + request.record_id,
                    "episode_count": len(result.graph.episodes),
                    "semantics_verified": False,
                    "physical_execution_tested": False,
                }
            raw = GraphSnapshot.model_validate_json(
                Path(request.graph.path).read_text()
            )
            progress_map = await run_blocking(
                project_map,
                raw,
                result.graph,
                request.versions,
                request.protocol.dev,
                cell_receipts=analysis_graph_receipts(request, raw),
                graph_id="draft:" + request.record_id,
                validation_conditions=request.protocol.validation.conditions
                if request.protocol.validation
                else (),
            )
            parent = policy_parent(
                progress_map,
                request.versions,
                request.protocol,
                purpose=request.purpose,
                current_versions=request.current_round_versions,
            )
            return {
                "structurally_valid": True,
                "graph_id": "draft:" + request.record_id,
                "episode_count": len(result.graph.episodes),
                "semantics_verified": False,
                "physical_execution_tested": False,
                "forecast_context": {
                    **forecast_context(progress_map, parent),
                    "references": [parent.version_id],
                    "author_wall_time_s": request.protocol.author_wall_time_s,
                    "meaning": "Prediction target for ONE author on this graph's scale. Supply its progress distribution and separate confidence, not conditional-outcome forecasts; the probability on 100 is its completion chance. Validation does not certify prediction accuracy.",
                },
            }
        if name == "read_graph":
            from reweave.evolution.contracts import (
                GraphSnapshot,
                SemanticGraph,
            )
            from reweave.evolution.evidence.trajectory import query_graph
            from reweave.evolution.graph.views import query_map, query_semantic

            options = dict(arguments)
            layer = options.pop("layer", "execution")
            if layer == "execution":
                graph = GraphSnapshot.model_validate(
                    self.references.read_json(
                        self.workspace / "context/execution_graph.json"
                    )
                )
                return query_graph(graph, **options)
            if any(key in options for key in ("first_frame", "last_frame", "features")):
                raise ValueError(
                    "frame and feature selection requires an execution episode"
                )
            if layer == "map":
                value = self.references.read_json(
                    self.workspace / "context/progress_map.json"
                )
                semantic = None
                if options.get("view") in {
                    "episode",
                    "transitions",
                    "witness",
                    "program",
                }:
                    payload = self.references.read_json(
                        self.workspace / "context/graph.json"
                    )
                    if payload is not None:
                        semantic = SemanticGraph.model_validate(payload)
                return query_map(value, graph=semantic, **options)
            if layer != "semantic":
                raise ValueError("unknown graph layer")
            payload = self.references.read_json(self.workspace / "context/graph.json")
            if payload is None:
                return {
                    "available": False,
                    "reason": "Initial evidence has not yet received a semantic graph.",
                }
            graph = SemanticGraph.model_validate(payload)
            value = self.references.read_json(
                self.workspace / "context/progress_map.json"
            )
            if value is not None and "interpretations" in value:
                identity = options.get("identity")
                for origin, raw_graph in value["interpretations"].items():
                    if (
                        identity is not None and identity.startswith(origin + "::")
                    ) or (
                        options.get("view") == "episode"
                        and identity in raw_graph["episodes"]
                    ):
                        graph = SemanticGraph.model_validate(raw_graph)
                        break
            return query_semantic(graph, **options)
        if name == "read_trace":
            result = await run_blocking(
                query_trace,
                self.trace_workspace,
                evidence_roots=self.evidence_roots,
                **arguments,
            )
            return result
        if name == "read_sensor_snapshot":
            from reweave.evolution.evidence.server import read_sensor_snapshot

            evidence_id = arguments["evidence_id"]
            root = self.evidence_roots.get(evidence_id)
            if root is None:
                raise ValueError(
                    "sensor episode is outside this request's admitted evidence"
                )
            result = await run_blocking(
                read_sensor_snapshot,
                root,
                evidence_id,
                arguments["observation_ref"],
                self.reports / "server_sensor_cache",
            )
            originals = {}
            for name in ("archive", "metadata"):
                record = artifact_record(Path(result[name + "_path"]))
                self.references.originals[record["path"]] = {
                    key: record[key] for key in ("artifact_id", "bytes")
                }
                originals[name] = self.references.reference(
                    record["path"], record["artifact_id"], record["bytes"]
                )
            return {**result, "originals": originals}
        if name == "read_video":
            from reweave.evolution.evidence.trace import read_video

            return await run_blocking(
                read_video, self.trace_workspace, self.evidence_roots, **arguments
            )
        if name == "read_original":
            from reweave.evolution.contracts import OriginalReference
            from reweave.evolution.evidence.trajectory import read_original

            original = OriginalReference.model_validate(arguments["reference"])
            path = Path(original.path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("original reference is missing or follows a link")
            for root in self.evidence_roots.values():
                if path.is_relative_to(
                    root
                ) and path.resolve() != root / path.relative_to(root):
                    raise ValueError(
                        "original reference follows a link outside its episode"
                    )
            image = await run_blocking(read_image_original, original)
            if image is not None:
                return {
                    "reference": self.references.reference(
                        original.path, original.artifact_id, original.bytes
                    ),
                    "selection_complete": True,
                    "_images": [image],
                }
            return {
                "value": await run_blocking(read_original, original),
                "selection_complete": True,
            }
        if name == "read_source":
            source_id = arguments["source_id"]
            relative = arguments["path"]
            source = self.executed_sources.get(source_id)
            if source is None or relative not in source["files"]:
                raise ValueError("unknown executed source file")
            root = Path(source["path"]).resolve()
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
                raise ValueError("executed source file escaped its inventory")
            raw = await run_blocking(path.read_bytes)
            return {
                "source_id": source_id,
                "path": relative,
                "bytes": len(raw),
                "source": raw.decode("utf-8"),
            }
        if name != "validate_candidate":
            raise ValueError("unknown development tool")
        if self.kind == "graph_analysis":
            raise ValueError("graph analysis has no candidate validation authority")
        return await validate_author_candidate(
            arguments,
            lambda **options: self._validate_code(case_results, **options),
        )

    async def _validate_code(self, case_results, *, interactions=False):
        from reweave.evolution.authoring.candidate import (
            candidate_root,
            output_inventory,
        )
        from reweave.evolution.infrastructure.async_work import run_blocking
        from reweave.evolution.validation.admission import admit_candidate

        root = candidate_root(self.workspace)
        before = await run_blocking(output_inventory, root)
        captured = {}
        bundle = await admit_candidate(
            self.workspace,
            require_plan=False,
            check_interactions=interactions,
            case_results=case_results,
            candidate_inventory=captured,
        )
        value = {
            "bundle_id": bundle.record_id,
            "contract_cases": sum(
                len(item.contract_cases)
                for item in (*bundle.workflows, bundle.context, bundle.loop)
            ),
            "contract_case_results": case_results,
        }
        after = await run_blocking(output_inventory, root)
        if not captured == before == after:
            raise ValueError(
                "candidate changed during validation; validate the settled files again"
            )
        return {
            **value,
            "phase": "interaction" if interactions else "code",
            "structurally_valid": True,
            "plan_checked": False,
            "candidate_files": after,
            "physical_execution_tested": False,
            "interaction_tested": interactions,
        }

    def preflight(
        self,
        *,
        readable: tuple[Path, ...],
        writable: tuple[Path, ...],
        control: Path,
        backend: str,
    ) -> None:
        """Exercise the real author boundary -> Host -> isolated worker path."""
        from reweave.evolution.authoring.workspace import (
            native_author_launcher,
            native_output,
        )

        launcher, env = native_author_launcher(
            self.workspace,
            native_output(self.workspace),
            control,
            backend=backend,
            cli=str(Path(sys.executable).resolve()),
            readable_roots=readable,
            writable_roots=writable,
        )
        client_source = self.workspace / "context/dev.py"
        if not client_source.is_file():
            raise ValueError("prepared Host client is missing")
        code = f"""
import os, pathlib, runpy, sys, tempfile
with tempfile.TemporaryDirectory() as temporary:
    lock = pathlib.Path(temporary) / '.lock.lock'
    lock.mkdir()
    lock.chmod(0o700)
    os.utime(lock, ns=(1000000000, 2000000000))
    assert lock.stat().st_mtime_ns == 2000000000
    with (lock / 'probe').open('w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        os.utime(stream.fileno(), None)
os.environ['REWEAVE_AUTHOR_TOOLS_URL'] = sys.argv[1]
client = runpy.run_path({str(client_source)!r})['call']
{("assert client('validate_candidate', phase='environment')['environment_ready']" if self.kind != "graph_analysis" else "assert client('read_graph', view='summary')['graph_id']")}
"""
        completed = subprocess.run(
            [launcher, "-c", code, self.server.url],
            cwd=self.workspace,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"author development preflight failed: {completed.stdout}\n{completed.stderr}"
            )
