"""Read-only source graph tools, plus checked topology proposals for shared authors."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from reweave.evolution.authoring.tools import (
    candidate_validation_tool,
    validate_author_candidate,
)
from reweave.evolution.contracts import EvidenceFile, OriginalReference, SemanticGraph
from reweave.evolution.distillation.topology import (
    SharedTopology,
    occurrence_key,
    revise_topology,
    validate_topology,
)
from reweave.evolution.graph.matching import semantic_query
from reweave.evolution.graph.retrieval import (
    GraphRetrieval,
    GraphRetrievalConfig,
    GraphSearch,
)


class TopologyMerge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: str
    right: str
    entity_mapping: dict[str, str]


class TopologyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: tuple[str, ...] = ()
    merges: tuple[TopologyMerge, ...] = ()
    reason: str = Field(min_length=1)


def apply_proposal(topology, graphs, raw):
    proposal = TopologyProposal.model_validate(raw)
    return revise_topology(
        topology,
        graphs,
        split=proposal.split,
        merges=[m.model_dump() for m in proposal.merges],
        reason=proposal.reason,
    )


class SharedAuthorTools:
    """No global corpus, live environment, inference probe or session inquiry."""

    def __init__(self, workspace, audit, *, default_task=None, validator=None):
        from mcp import types
        from mcp.server.lowlevel import Server

        from reweave.planner.utils.http_mcp_server import HttpMcpServer

        self.workspace = workspace
        self.audit = audit
        self.default_task = default_task
        self.validator = validator
        self.graphs = {
            name: SemanticGraph.model_validate(value)
            for name, value in json.loads(
                (workspace / "context/source_graphs.json").read_text()
            ).items()
        }
        fragment_path = workspace / "context/fragment_transfer.json"
        self.fragment_report = (
            json.loads(fragment_path.read_text()) if fragment_path.exists() else None
        )
        gap_path = workspace / "context/student_gaps.json"
        self.student_gaps = None
        self.teacher_units = {}
        self.query_units = {}
        if gap_path.exists():
            self.student_gaps = json.loads(gap_path.read_text())
            self.teacher_units = json.loads(
                (workspace / "context/teacher_unit_bank.json").read_text()
            )["units"]
            self._validate_student_gaps()
        self.retrieval = None
        if self.graphs:
            config = GraphRetrievalConfig(
                graphs={
                    name: (
                        workspace / "context/source_graphs" / f"{name}.json"
                    ).resolve()
                    for name in self.graphs
                },
                cache=(audit.parent / "source_catalog.sqlite").resolve(),
                solver_seconds=json.loads(
                    (workspace / "context/input.json").read_text()
                )["alignment_seconds"],
            )
            self.retrieval = GraphRetrieval(config, admitted_graphs=self.graphs)
        self.originals = self.retrieval.originals() if self.retrieval else {}
        self.topology = None
        path = workspace / "context/shared_topology.json"
        if path.exists():
            self.topology = SharedTopology.model_validate_json(path.read_text())
            validate_topology(self.topology, self.graphs)
        self.lock = asyncio.Lock()
        app = Server("shared_sources", version="1.0.0")

        @app.list_tools()
        async def catalog():
            tools = [
                types.Tool(
                    name="retrieve_graphs",
                    description="Autonomously query admitted source graphs. Explicit query supports local conditions, an edge or a complete graph. query_task selects a complete source graph; no query uses the current task only when one is defined. Shared consolidation has no distinguished target. Similarity is not evidence of physical transfer.",
                    inputSchema={
                        **GraphSearch.model_json_schema(),
                        "properties": {
                            **GraphSearch.model_json_schema()["properties"],
                            "query_task": {"type": "string"},
                        },
                    },
                ),
                types.Tool(
                    name="read_reference_graph",
                    description="Read source conditions, nodes, transitions, witness/program references, or the shared fluent catalog. Omit task to list admitted sources. Catalog lookup text is optional and uses the configured local encoder only.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "view": {
                                "type": "string",
                                "enum": [
                                    "summary",
                                    "conditions",
                                    "node",
                                    "transitions",
                                    "program",
                                    "catalog",
                                ],
                            },
                            "identity": {"type": "string"},
                            "text": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                        "additionalProperties": False,
                    },
                ),
                types.Tool(
                    name="read_original",
                    description="Read only originals already referenced by admitted source graphs. Complete bundles and JSON pointers expose coupled C/W/L dependencies; image originals retain their original pixels.",
                    inputSchema=OriginalReference.model_json_schema(),
                ),
            ]
            if self.fragment_report is not None:
                tools = [tool for tool in tools if tool.name != "retrieve_graphs"]
                tools.append(
                    types.Tool(
                        name="read_fragment_reuse",
                        description="Read paged source-local adaptation suggestions. Use query_task and edge to focus a local need; next_offset accesses remaining suggestions. Strict substitution is diagnostic only.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query_task": {"type": "string"},
                                "reference_task": {"type": "string"},
                                "edge": {"type": "string"},
                                "offset": {"type": "integer", "minimum": 0},
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                            },
                            "additionalProperties": False,
                        },
                    )
                )
            if not self.graphs:
                tools = []
            if self.student_gaps is not None:
                tools.append(
                    types.Tool(
                        name="query_student_gap",
                        description="Read current source-development gap mechanisms, all candidate units, or matching variants and original interface details. Pages preserve all candidates without ranking; next_offset reaches the remainder. Omit gap_id for mechanisms, or provide unit alone to inspect any admitted teacher interface. Correspondence is synthesis guidance, not executable transfer evidence.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "gap_id": {"type": "string", "minLength": 1},
                                "unit": {"type": "string", "minLength": 1},
                                "offset": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "default": 0,
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 64,
                                    "default": 8,
                                },
                            },
                            "additionalProperties": False,
                        },
                    )
                )
            if self.validator is not None:
                tools.append(candidate_validation_tool())
            if self.topology is not None:
                tools.append(
                    types.Tool(
                        name="validate_topology",
                        description="Check proposed splits/role-aware merges against all source conditions, dependencies and witnesses. Does not certify task success. Write the final proposal to .scratch/topology_proposal.json for Host admission.",
                        inputSchema=TopologyProposal.model_json_schema(),
                    )
                )
            return tools

        self.catalog = catalog

        @app.call_tool()
        async def call(name, arguments):
            async with self.lock:
                started = time.monotonic()
                entry = {"tool": name, "arguments": arguments}
                try:
                    value = await self.model_call(name, arguments or {})
                    images = value.pop("_images", [])
                    entry["ok"] = True
                    return types.CallToolResult(
                        structuredContent=value,
                        content=[
                            types.TextContent(type="text", text=json.dumps(value)),
                            *(types.ImageContent(**item) for item in images),
                        ],
                    )
                except Exception as error:
                    entry.update(ok=False, error=str(error))
                    return types.CallToolResult(
                        isError=True,
                        structuredContent={"error": str(error)},
                        content=[types.TextContent(type="text", text=str(error))],
                    )
                finally:
                    entry["elapsed_s"] = time.monotonic() - started
                    self.audit.parent.mkdir(parents=True, exist_ok=True)
                    with self.audit.open("a") as stream:
                        stream.write(json.dumps(entry) + "\n")

        self.server = HttpMcpServer(app, path="/mcp/" + secrets.token_urlsafe(24))

    def _validate_student_gaps(self):
        """Admit only literal interfaces and queries from the supplied sources."""
        from reweave.evolution.distillation.student_gaps import teacher_unit_bank

        self.query_units = teacher_unit_bank(self.graphs)["units"]
        try:
            for identity, unit in self.teacher_units.items():
                task, edge = unit["task"], unit["edge"]
                if (
                    task not in self.graphs
                    or edge not in self.graphs[task].edges
                    or identity != occurrence_key(task, edge)
                ):
                    raise ValueError("teacher unit is outside admitted sources")
            report = self.student_gaps
            mechanisms = {m["id"]: m for m in report["mechanisms"]}
            queries = {g["id"]: g for g in report["gaps"]}
            if len(mechanisms) != len(report["mechanisms"]) or len(queries) != len(
                report["gaps"]
            ):
                raise ValueError("duplicate student gap identity")
            if set(report["matches"]) - mechanisms.keys():
                raise ValueError("unknown student gap mechanism")
            for identity, mechanism in mechanisms.items():
                task = mechanism["task"]
                if task not in self.graphs:
                    raise ValueError("student gap is outside admitted sources")
                graph = self.graphs[task]
                for name in ("retained_node", "frontier_node"):
                    node = mechanism[name]
                    if node is not None and node not in graph.nodes:
                        raise ValueError("student gap names an unknown source node")
                candidates = mechanism["candidate_units"]
                if len(set(candidates)) != len(candidates) or not set(candidates) <= (
                    self.teacher_units.keys()
                ):
                    raise ValueError("candidate unit is outside admitted sources")
                for query in mechanism["query_ids"]:
                    gap = queries[query]
                    if (
                        gap["mechanism"] != identity
                        or self.query_units[gap["unit"]]["task"] != task
                        or gap["record_id"] not in mechanism["record_ids"]
                    ):
                        raise ValueError("student gap query differs from its mechanism")
                for match in report["matches"].get(identity, []):
                    if (
                        match["query"] not in mechanism["query_ids"]
                        or match["unit"] not in candidates
                        or match["match_kind"] not in {"full", "partial"}
                    ):
                        raise ValueError(
                            "student gap match is outside admitted queries"
                        )
            if any(
                g["mechanism"] not in mechanisms
                or g["id"] not in mechanisms[g["mechanism"]]["query_ids"]
                for g in queries.values()
            ):
                raise ValueError("unassigned student gap query")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("malformed source student gap report") from error

    def query_student_gap(self, arguments):
        if self.student_gaps is None:
            raise ValueError("student gap tool unavailable")
        if not isinstance(arguments, dict):
            raise ValueError("invalid student gap query arguments")
        if set(arguments) - {"gap_id", "unit", "offset", "limit"}:
            raise ValueError("unknown student gap query argument")
        for key in ("gap_id", "unit"):
            if key in arguments and (
                not isinstance(arguments[key], str) or not arguments[key]
            ):
                raise ValueError("invalid student gap query identity")
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 8)
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 64
        ):
            raise ValueError("invalid student gap page")

        def page(rows):
            end = min(offset + limit, len(rows))
            return {
                "total": len(rows),
                "offset": offset,
                "next_offset": end if end < len(rows) else None,
                "complete": end >= len(rows),
            }, rows[offset:end]

        report = self.student_gaps
        gap_id, unit = arguments.get("gap_id"), arguments.get("unit")
        if unit is not None and unit not in self.query_units:
            raise ValueError("unit is outside admitted sources")
        if gap_id is None:
            if unit is not None:
                return {"unit": unit, "detail": self.query_units[unit]}
            rows = [
                {
                    **{
                        k: m[k]
                        for k in (
                            "id",
                            "task",
                            "retained_node",
                            "frontier_node",
                            "record_ids",
                            "query_ids",
                        )
                    },
                    "candidate_count": len(m["candidate_units"]),
                }
                for m in report["mechanisms"]
            ]
            metadata, rows = page(rows)
            return {
                "student_bundle_id": report["student_bundle_id"],
                "mechanisms": rows,
                "summary": report["summary"],
                **metadata,
            }
        mechanism = next((m for m in report["mechanisms"] if m["id"] == gap_id), None)
        if mechanism is None:
            raise ValueError("unknown student gap mechanism")
        matches = report["matches"].get(gap_id, [])
        if unit is not None:
            query_units = {
                g["unit"] for g in report["gaps"] if g["mechanism"] == gap_id
            }
            if unit not in mechanism["candidate_units"] and unit not in query_units:
                raise ValueError("unit is outside this student gap")
            variants = [m for m in matches if m["unit"] == unit]
            metadata, rows = page(variants)
            return {
                "gap_id": gap_id,
                "unit": unit,
                "detail": self.query_units[unit],
                "matches": rows,
                "not_a_match": not variants,
                **metadata,
            }
        rows = []
        for identity in mechanism["candidate_units"]:
            detail = self.teacher_units[identity]
            variants = [m for m in matches if m["unit"] == identity]
            rows.append(
                {
                    "unit": identity,
                    "task": detail["task"],
                    "edge": detail["edge"],
                    "variant_count": len(variants),
                    "full_count": sum(m["match_kind"] == "full" for m in variants),
                    "partial_count": sum(
                        m["match_kind"] == "partial" for m in variants
                    ),
                    "entry_flag_counts": {
                        key: sum(bool(m.get(key)) for m in variants)
                        for key in ("missing_entry", "entry_conflicts", "unbound_entry")
                    },
                    "program_ids": detail.get("program_ids", []),
                }
            )
        metadata, rows = page(rows)
        return {
            "gap_id": gap_id,
            "queries": [g for g in report["gaps"] if g["mechanism"] == gap_id],
            "candidates": rows,
            **metadata,
        }

    async def model_call(self, name, arguments):
        from reweave.evolution.infrastructure.async_work import run_blocking

        if name == "validate_candidate" and self.validator is not None:
            return await validate_author_candidate(arguments, self.validator)
        if name == "query_student_gap":
            return self.query_student_gap(arguments)
        if name == "read_fragment_reuse" and self.fragment_report is not None:
            if set(arguments) - {
                "query_task",
                "reference_task",
                "edge",
                "offset",
                "limit",
            }:
                raise ValueError("unknown fragment query argument")
            for key in ("query_task", "reference_task"):
                if key in arguments and arguments[key] not in self.graphs:
                    raise ValueError("fragment query is outside admitted sources")
            offset, limit = arguments.get("offset", 0), arguments.get("limit", 20)
            if (
                type(offset) is not int
                or offset < 0
                or type(limit) is not int
                or not 1 <= limit <= 100
            ):
                raise ValueError("invalid fragment page")
            inventory = self.fragment_report["inventory"]
            proposals = [
                p
                for p in self.fragment_report.get(
                    "adaptation_proposals", self.fragment_report["proposals"]
                )
                if (
                    "query_task" not in arguments
                    or inventory[p["query"]]["task"] == arguments["query_task"]
                )
                and (
                    "reference_task" not in arguments
                    or inventory[p["reference"]]["task"] == arguments["reference_task"]
                )
                and (
                    "edge" not in arguments
                    or inventory[p["query"]]["edge"] == arguments["edge"]
                )
            ]
            end = min(offset + limit, len(proposals))
            return {
                "proposals": proposals[offset:end],
                "total": len(proposals),
                "offset": offset,
                "next_offset": end if end < len(proposals) else None,
                "complete": end >= len(proposals),
                "callable": False,
                "guidance_metrics": self.fragment_report.get("guidance_metrics"),
            }
        if name == "retrieve_graphs" and self.fragment_report is not None:
            raise ValueError("use read_fragment_reuse for the fragment method")
        if self.retrieval is None:
            raise ValueError("tool unavailable in source-only shared author")
        if name == "retrieve_graphs":
            arguments = dict(arguments)
            task = arguments.pop("query_task", self.default_task)
            request = GraphSearch.model_validate(arguments)
            if self.student_gaps is not None:
                reference_tasks = {u["task"] for u in self.teacher_units.values()}
                request = request.model_copy(
                    update={
                        "exclude_tasks": tuple(
                            sorted(
                                set(request.exclude_tasks)
                                | (self.graphs.keys() - reference_tasks)
                            )
                        )
                    }
                )
            if task is not None and task not in self.graphs:
                raise ValueError("query_task is outside admitted sources")
            default = semantic_query(self.graphs[task])[0] if task is not None else None
            return await run_blocking(
                self.retrieval.search, request, default_query=default
            )
        if name == "read_reference_graph":
            if set(arguments) - {"task", "view", "identity", "text", "top_k"}:
                raise ValueError("unknown reference read argument")
            task, view = arguments.get("task"), arguments.get("view", "summary")
            if task is None and view == "catalog":
                return await run_blocking(
                    self.retrieval.read_catalog,
                    text=arguments.get("text"),
                    top_k=arguments.get("top_k", 10),
                )
            if task is None:
                return {
                    "tasks": [
                        {"task": task, "nodes": len(g.nodes), "edges": len(g.edges)}
                        for task, g in self.graphs.items()
                    ]
                }
            return self.retrieval.read(
                task, view=view, identity=arguments.get("identity")
            )
        if name == "read_original":
            original = OriginalReference.model_validate(arguments)
            metadata = self.originals.get(original.path)
            if metadata != {
                "artifact_id": original.artifact_id,
                "bytes": original.bytes,
            }:
                raise ValueError("original is outside admitted source evidence")
            EvidenceFile(path=original.path, **metadata).verify()
            from reweave.evolution.authoring.tools import read_image_original

            image = read_image_original(original)
            if image:
                return {"_images": [{k: v for k, v in image.items() if k != "type"}]}
            path = Path(original.path)
            if original.pointer:
                value = json.loads(path.read_text())
                for part in original.pointer.lstrip("/").split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    value = value[int(key)] if isinstance(value, list) else value[key]
                return {"value": value}
            return {"text": path.read_text()}
        if name == "validate_topology":
            if self.topology is None:
                raise ValueError("no shared topology is configured")
            result = apply_proposal(self.topology, self.graphs, arguments)
            return {
                "valid": True,
                "source_constraints_preserved": True,
                "revision": result.revision,
                "mechanisms": len(result.mechanisms),
                "shared_mechanisms": sum(
                    len(m.members) > 1 for m in result.mechanisms.values()
                ),
            }
        raise ValueError("tool unavailable in source-only shared author")


class ImplementationSite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pointers: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class SharedImplementation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_constraints: dict[str, ImplementationSite] = Field(
        description="EXACTLY one entry per source task key in shared_topology.sources; no per-condition, fluent, entity or other additional keys. Each entry locates that source's retained constraints in deployed C/W/L."
    )
    interfaces: dict[str, ImplementationSite] = Field(
        description="EXACTLY one entry per literal source-interface member key in shared_topology.mechanisms; no mechanism-group IDs, condition IDs or additional keys. Each entry locates the interface in deployed C/W/L."
    )


def implementation_schema(topology):
    """Expose the same exact key sets required by Host implementation admission."""
    schema = SharedImplementation.model_json_schema()
    expected = {
        "source_constraints": sorted(topology.sources),
        "interfaces": sorted(
            {m.key for group in topology.mechanisms.values() for m in group.members}
        ),
    }
    for name, keys in expected.items():
        field = schema["properties"][name]
        field["propertyNames"] = {"enum": keys}
        field["required"] = keys
    return schema


def _implementation_key_error(name, actual, expected, missing_message):
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if not missing and not extra:
        return
    details = {
        "field": name,
        "missing_count": len(missing),
        "missing_keys_sample": missing[:8],
        "extra_count": len(extra),
        "extra_keys_sample": extra[:8],
        "required_keys": sorted(expected),
    }
    message = missing_message if missing else "implementation has unexpected keys"
    raise ValueError(
        f"{message}; {name} requires EXACTLY required_keys (samples show at most 8 keys): "
        + json.dumps(details, separators=(",", ":"))
    )


def validate_implementation(raw, topology, bundle):
    """Require executable locations for every source contract, never semantic proof."""
    result = SharedImplementation.model_validate(raw)
    _implementation_key_error(
        "source_constraints",
        set(result.source_constraints),
        set(topology.sources),
        "implementation omits source-specific constraints",
    )
    expected = {m.key for group in topology.mechanisms.values() for m in group.members}
    _implementation_key_error(
        "interfaces",
        set(result.interfaces),
        expected,
        "implementation omits a source interface",
    )
    data = bundle.model_dump(mode="json")
    for site in (*result.source_constraints.values(), *result.interfaces.values()):
        for pointer in site.pointers:
            if not pointer.startswith("/") or pointer.split("/")[1] not in {
                "context",
                "workflows",
                "loop",
            }:
                raise ValueError("implementation pointers must select deployed C/W/L")
            value = data
            try:
                for part in pointer[1:].split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    value = value[int(key)] if isinstance(value, list) else value[key]
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError("implementation pointer does not resolve") from error
            if not isinstance(value, (str, list, dict)) or not value:
                raise ValueError("implementation pointer selects no executable content")
    return result
