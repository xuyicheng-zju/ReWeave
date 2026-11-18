"""Isolated native Codex author for a single compact shared C/W/L student."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

from reweave.evolution.authoring.author import AUTHOR_VALIDATION_ERRORS, NativeSession
from reweave.evolution.authoring.candidate import (
    capture_bundle,
    output_inventory,
    write_bundle_source,
)
from reweave.evolution.authoring.fixtures import SUITE, seed_parent_interactions
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import (
    EvaluationConditions,
    EvaluationRecord,
    SemanticGraph,
    Usage,
)
from reweave.evolution.distillation.engine import bind_json
from reweave.evolution.distillation.tools import (
    SharedAuthorTools,
    TopologyProposal,
    apply_proposal,
    implementation_schema,
    validate_implementation,
)
from reweave.evolution.distillation.topology import SharedTopology, validate_topology
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.graph.matching import match_graphs, semantic_query
from reweave.evolution.infrastructure.campaign_slots import offline_admission
from reweave.evolution.infrastructure.native_retry import (
    NativeTransportFailure,
    is_retryable_native_error,
)
from reweave.evolution.infrastructure.session import serialize_native_event
from reweave.evolution.validation.admission import (
    _read_workspace_file,
    check_interaction_suite,
    check_runtime_contracts,
)
from reweave.utils.persistence import atomic_json, atomic_text

FRAGMENT_INSTRUCTIONS = """When context/input.json provides fragment_transfer, use read_fragment_reuse
with query_task and edge to read paged adaptation_proposals; follow next_offset
when needed. Do not dump the entire report into context. A query is a local need within an admitted SOURCE
teacher, never an unseen evaluation task. Shared local results anchor suggestions;
full-interface substitution is only a diagnostic in proposals. Use matched_results,
uncovered_results, entry_additions, entry_conflicts and unbound_reference_entry to
adapt source code. Bind only the roles in entity_mapping; other roles are unresolved.
Conflicts are code/context adaptation issues, not instructions to alter immutable
object properties. These suggestions are not executable guarantees.
Preserve all original entry conditions, outcomes, evidence and failed examples.
Combine useful adaptation suggestions for the same query in ONE shared C/W/L
implementation. Legacy missing_entry/completion_options are optional strict diagnostics. Resolve object bindings and possible side effects
in the actual code. No standalone call is admitted by a symbolic match. Keep
unmatched source capabilities. Physical source-development evaluation determines
whether the resulting integration works; synthetic checks only validate software.
"""

INSTRUCTIONS = """Construct ONE compact cross-task executable harness, not a library of
complete task specialists. Read context/input.json and the admitted source evidence.
Refactor coupled C/W/L mechanisms into common implementations; preserve only
necessary branches on observable physical conditions. Do not dispatch on benchmark
task IDs, memorized task names, reset IDs or evaluator outcomes. Do not embed whole
teacher harnesses behind a router. You may discover common mechanisms yourself.
Balance task success, teacher capability retention and delta Harness size under
the Host criteria. Delta size counts added/replacement UTF-8 source/text/schema
lines against context/input.json's fixed size_reference, not the current parent
and not net total-size growth. Unchanged base content costs zero; removals never
cancel new content. Removing essential capability is not compression.
The complexity term is beta times (shared delta kB minus mean teacher delta kB).
The teacher mean and H0 reference are fixed for the run. With
criteria.fixed_topology_scoring, the Host uses the POSITIVE PART of this size
excess: L=(1-source_SR)+topology_weight*(1-mean_retained_topology_progress)
+beta*max(0, shared_delta_kB-mean_teacher_delta_kB). Compression cannot cancel
task failure. Reference topologies stay fixed; the Host independently projects
new trajectories, with official success scoring 1 and necessary retained
milestones providing partial credit. Optional branches and retries add no credit.
Otherwise the legacy complexity term is signed and may make total loss negative.
{{continual_guidance}}
For min_loss selection, the Host stops only when total loss is strictly below
criteria.loss_threshold, or the mutation-round budget is exhausted. Old-task
degradation is already represented in the objective; no separate retention veto
applies to selection or stopping.
All-source success alone does not stop consolidation: continue reducing local
capability loss and redundant deployed C/W/L while preserving source performance.
When criteria.teacher_retention is false, there is no teacher-development cutoff:
optimize source development success, configured topology progress and delta size
on all declared development conditions. When criteria.sequential_updates is true,
edit the supplied immediate previous Harness. When criteria.checkpoint_selection
is min_loss, the Host selects the lowest full development loss among H0 and all
evaluated updates, including complexity, and freezes that checkpoint without
another author update. Otherwise historical sequential plans return the last
evaluated Harness.
Ten-seed specialist degradation is measured only after the shared Harness freezes;
never request those test outcomes or use them to revise a candidate.
The base model, runtime and tool contracts are fixed. No physical actions, network
experiments, model probes, session inquiry or external evidence gathering.
Edit candidate/ using the same complete source layout as the source examples.
Author .scratch/tests/interaction.json with production-path synthetic scenarios
covering changed C/W/L, each changed workflow, and recovery through a subsequent
Planner decision when C/L with workflows changes. Fixtures check interfaces only;
only subsequent Host physical evaluations establish task success. Return a complete
candidate even when uncertain, and describe assumptions in candidate/SKILL.md.
{{topology_guidance}}
"""

CONTINUAL_INSTRUCTIONS = "Continual updates also use the positive part of size excess; compression never\noffsets task failure. In continual mode, update ONE shared C/W/L directly on all\ndeclared old-task and new-task development conditions. Do not train separate new\ntask specialists or add a later merge stage. Old-task topologies are immutable;\nthe Host refreshes the two new-task topologies from each evaluated shared version.\nTheir current frontiers query only the old-task mechanism bank. Use gap_index\nand query_student_gap to inspect useful matches, and retain source conditions and\nunmatched capabilities when adapting mechanisms into the same shared code.\n"

TOPOLOGY_INSTRUCTIONS = """When context/shared_topology.json exists, it is a lossless source factorization:
read BOTH common mechanisms AND every source residual/qualification/dependency.
Do not replace source-specific guards with the intersection of common conditions.
Provide .scratch/shared_implementation.json using its context schema, locating
all source constraints and literal interfaces in actual deployed C/W/L pointers.
The source_constraints keys are EXACTLY the admitted source task names; interfaces
keys are EXACTLY the literal occurrence keys enumerated in that schema. Do not
add per-condition keys to either mapping. Once the manifest is written, rerun
validate_candidate phase=interaction to check its keys and executable pointers
along with the interactions before returning.
You may propose splits or role-aware merges in .scratch/topology_proposal.json.
Host reconstructs ALL source graphs to admit a proposal. This checks encoded
constraints, not physical correctness; subsequent full-source execution decides.
Source graph tools remain available throughout each author turn: retrieve_graphs,
read_reference_graph, read_original, and validate_topology when configured.
Read the catalog before encoding an explicit local/whole query. query_task selects
one admitted source as the query. Consolidation has no default target graph.
Read original programs and witness evidence on demand; all necessary conditions
and coupled dependencies must survive integration. No held-out evidence is allowed.
For method=aligned, use the explicit source correspondences and shared topology
to distill teacher mechanisms, rather than solving the tasks afresh. Compare
aligned entry/result conditions to identify which requirements are already
supported and which remain source-specific. Seek complementary teacher mechanisms
for missing conditions; integrate them with consistent observable role bindings
in the shared C/W/L. Use source-local retrieval when whole-graph correspondences
are sparse. Record the teacher implementations, role correspondences and remaining
conditions for each changed mechanism in candidate/SKILL.md. Preserve every source
residual, including unmatched capabilities. A partial/unfinished alignment is a
synthesis suggestion; its score does not certify equivalence or execution.
When context/input.json provides gap_index, begin with that CURRENT student's
observed gaps. query_student_gap reads the compact mechanism index, candidate
summaries, individual bindings and complete teacher-interface details. Default
pages contain eight entries; follow next_offset when useful. This is navigation,
not top-k selection. Full and partial result matches are references, not a ranking
or executable plan. Compare code and evidence yourself; choose useful references,
combine them, or decline them. No minimum-cover plan or required reference count
is imposed. Keep each case's known facts, entry differences and role bindings
separate. Alternative incoming interfaces are alternatives, not simultaneous
requirements or proven failure causes. Unknown entry is not a physical deficit.
Use full teacher C/W/L when needed. Repair the preceding student while preserving
protected_outcomes; avoid rebuilding unrelated working mechanisms. The parent's
interface-to-code map is provided by parent_implementation and seeded in
.scratch/shared_implementation.json. In SKILL.md, record gaps addressed, teacher
unit IDs actually used or declined, and deployed code locations. Do not claim
success from a retrieval match. Overall official_success is never an operation
correspondence. No held-out feedback is allowed.

"""


class NativeSharedAuthor(NativeSession):
    """Reuse native session, OS read boundary, transcripts and admission checks.

    No general archive server or session inquiry is exposed. Only exact source
    evidence files are readable; the shared run's test directory is never granted.
    """

    source_queries = True
    instructions = INSTRUCTIONS

    def _instructions(self, request):
        guidance = "" if request["method"] == "direct" else TOPOLOGY_INSTRUCTIONS
        return self.instructions.replace("{{topology_guidance}}", guidance).replace(
            "{{continual_guidance}}",
            CONTINUAL_INSTRUCTIONS if request["mode"] == "continual" else "",
        ) + (
            FRAGMENT_INSTRUCTIONS
            if request.get("fragment_transfer") is not None
            else ""
        )

    prompt = "Read context/input.json and context/feedback_focus.json and distill the supplied teachers' C/W/L mechanisms into the preceding shared student. Preserve mechanisms already integrated; use current source development failures and loss terms to target missing or failing teacher mechanisms instead of rebuilding the student. All teachers and historical source evidence remain available; read their implementations and older trajectories when needed for the affected mechanisms. Briefly identify mechanisms retained or adapted in candidate/SKILL.md. Reuse the parent interaction fixtures when provided and update affected coverage. Read context/runtime.schema.json: .scratch/tests/interaction.json must be a nonempty array of interaction_scenario objects. Run validate_candidate phase=interaction before returning, and repair its failures in this same turn. For w/o topology consolidation, use teacher code and raw evidence without graph guidance; no alignment, source-constraint manifest or topology proposal is required. Do not access held-out evidence."

    def _codex_experiment_instructions(self, workspace, directory):
        return "\nOnly the current source workspace is authorized. No external probes."

    def prepare(self, request, directory):
        if (
            self.source_queries
            and request["method"] == "direct"
            and (
                request.get("source_graphs")
                or any(t["graph"] is not None for t in request["teachers"])
                or any(
                    request.get(key)
                    for key in (
                        "shared_topology",
                        "fixed_alignments",
                        "mechanisms",
                        "conditional_anchors",
                        "student_gaps",
                        "teacher_unit_bank",
                        "topology_refresh",
                        "fragment_transfer",
                    )
                )
                or any(
                    "topology_records" in item["score"] for item in request["history"]
                )
            )
        ):
            raise ValueError(
                "w/o topology author cannot receive graph guidance or projected witness details"
            )
        bind_json(directory / "request.json", request)
        bind_json(directory / "policy.json", self.policy.model_dump(mode="json"))
        if self.policy.author != "codex" or self.policy.graph_retrieval is not None:
            raise ValueError(
                "shared author requires Codex with a source-local corpus, not a global retrieval configuration"
            )
        if self.runtime_manifest is None:
            raise ValueError("shared author requires an active runtime manifest")
        workspace = directory / "workspace"
        context = workspace / "context"
        readable = []
        conditions = {
            json.dumps([i["task"], i["initialization"]], sort_keys=True)
            for task in request["tasks"]
            for i in task["dev"]
        }
        evidence = []
        seen = set()
        for raw in [r for t in request["teachers"] for r in t["evidence"]] + request[
            "feedback"
        ]:
            record = EvaluationRecord.model_validate(raw)
            key = json.dumps(
                [record.request.instance.task, record.request.instance.initialization],
                sort_keys=True,
            )
            if record.request.split != "dev" or key not in conditions:
                raise ValueError("native shared author received non-source evidence")
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            evidence.append(raw)
            for item in record.evidence:
                item.verify()
                path = Path(item.path)
                if not path.is_file() or path.resolve() != path.absolute():
                    raise ValueError(
                        "source evidence must be a regular unlinked original"
                    )
                readable.append(path.resolve())
        source_graphs = {
            name: SemanticGraph.model_validate(value)
            for name, value in request.get("source_graphs", {}).items()
        }
        for teacher in request["teachers"]:
            if teacher["graph"] is not None:
                graph = SemanticGraph.model_validate(teacher["graph"])
                if (
                    teacher["task"] in source_graphs
                    and source_graphs[teacher["task"]] != graph
                ):
                    raise ValueError("source graph changed between evidence channels")
                source_graphs[teacher["task"]] = graph
        if not source_graphs.keys() <= {task["name"] for task in request["tasks"]}:
            raise ValueError("native shared graph bank contains a non-source task")
        topology = None
        if request.get("shared_topology") is not None:
            topology = SharedTopology.model_validate(request["shared_topology"])
            validate_topology(topology, source_graphs)
        # Navigation only: retain every original and teacher, but start diagnosis
        # with the latest observed result for each declared source condition.
        latest = {}
        for index, raw in enumerate(evidence):
            instance = raw["request"]["instance"]
            key = json.dumps(
                [instance["task"], instance["initialization"]], sort_keys=True
            )
            latest[key] = {
                "record_id": raw["record_id"],
                "evidence_pointer": f"context/evidence.json#/{index}",
                "official_success": raw["official_success"],
                "failure_origin": raw["failure_origin"],
            }
        focus = {
            "latest_source_conditions": list(latest.values()),
            "loss_history": request["history"],
            "teachers": "context/teachers",
            "complete_evidence": "context/evidence.json",
            "meaning": "Start with current failures and loss terms. Teachers and all earlier source evidence remain available on demand; this index does not discard evidence or establish causality.",
        }
        if request.get("student_gaps") is not None:
            focus["student_gaps"] = "context/gap_index.json"
            focus["gap_summary"] = request["student_gaps"]["summary"]
        if workspace.exists():
            if not (workspace / "prepared.json").is_file():
                raise ValueError(
                    "incomplete source workspace; preserve it for recovery"
                )
            atomic_json(
                context / "runtime.schema.json",
                self.runtime_schema(request["conditions"]["controller_request_unit"]),
            )
            atomic_json(context / "feedback_focus.json", focus)
            if topology is not None:
                atomic_json(
                    context / "shared_implementation.schema.json",
                    implementation_schema(topology),
                )
            return workspace, tuple(readable)
        staging = directory / "preparation"
        if staging.exists():
            os.replace(staging, directory / f"preparation-interrupted-{time.time_ns()}")
        workspace = staging
        context = workspace / "context"
        context.mkdir(parents=True)
        parent = AgentBundle.model_validate(request["parent"])
        write_bundle_source(parent, workspace / "candidate")
        atomic_text(workspace / "candidate/SKILL.md", "")
        write_bundle_source(parent, context / "parent")
        if request.get("size_reference") is not None:
            size_reference = AgentBundle.model_validate(request["size_reference"])
            write_bundle_source(size_reference, context / "size_reference")
        graphs = []
        for index, teacher in enumerate(request["teachers"]):
            source = context / "teachers" / f"{index:03d}"
            bundle = AgentBundle.model_validate(teacher["bundle"])
            write_bundle_source(bundle, source)
            atomic_json(
                source / "provenance.json",
                {"task": teacher["task"], "learned_tasks": teacher["learned_tasks"]},
            )
            # All arms retain teacher code and raw evidence. Graph guidance is
            # admitted only by the topology-enabled synthesis/analysis contract.
            if teacher["graph"] is not None:
                graph = SemanticGraph.model_validate(teacher["graph"])
                atomic_json(source / "graph.json", graph.model_dump(mode="json"))
                if graph.matching is not None and request["method"] != "fragments":
                    graphs.append(
                        (
                            teacher["task"],
                            semantic_query(graph)[0],
                        )
                    )
        alignments = []
        if request["method"] == "aligned" and not request.get("fixed_alignments"):
            for query_name, query in graphs:
                for reference_name, reference in graphs:
                    if query_name == reference_name:
                        continue
                    alignments.append(
                        {
                            "query": query_name,
                            "reference": reference_name,
                            "alignment": match_graphs(
                                query, reference, seconds=request["alignment_seconds"]
                            ),
                        }
                    )
        atomic_json(
            context / "alignments.json", request.get("fixed_alignments") or alignments
        )
        atomic_json(
            context / "source_graphs.json",
            {
                name: graph.model_dump(mode="json")
                for name, graph in source_graphs.items()
                if graph.matching is not None
            },
        )
        for name, graph in source_graphs.items():
            if graph.matching is not None:
                atomic_json(
                    context / "source_graphs" / f"{name}.json",
                    graph.model_dump(mode="json"),
                )
        if topology is not None:
            atomic_json(
                context / "shared_topology.json", topology.model_dump(mode="json")
            )
            atomic_json(
                context / "topology_proposal.schema.json",
                TopologyProposal.model_json_schema(),
            )
            atomic_json(
                context / "shared_implementation.schema.json",
                implementation_schema(topology),
            )
        if request.get("fragment_transfer") is not None:
            atomic_json(
                context / "fragment_transfer.json", request["fragment_transfer"]
            )
        if request.get("student_gaps") is not None:
            if request["student_gaps"]["student_bundle_id"] != parent.record_id:
                raise ValueError("gap report differs from the supplied student")
            atomic_json(context / "student_gaps.json", request["student_gaps"])
            gap_index = {
                k: request["student_gaps"][k]
                for k in (
                    "schema",
                    "student_bundle_id",
                    "summary",
                    "protected_outcomes",
                    "unresolved",
                )
            }
            gap_index["mechanisms"] = [
                {
                    **{k: v for k, v in m.items() if k != "candidate_units"},
                    "candidate_count": len(m["candidate_units"]),
                }
                for m in request["student_gaps"]["mechanisms"]
            ]
            atomic_json(context / "gap_index.json", gap_index)
            atomic_json(
                context / "teacher_unit_bank.json", request["teacher_unit_bank"]
            )
            for path in sorted(directory.parent.parent.glob("*/author/result.json")):
                previous = json.loads(path.read_text())
                if (
                    previous["bundle"]["record_id"] == parent.record_id
                    and previous.get("implementation") is not None
                ):
                    atomic_json(
                        context / "parent_implementation.json",
                        previous["implementation"],
                    )
                    atomic_json(
                        workspace / ".scratch/shared_implementation.json",
                        previous["implementation"],
                    )
                    break

        if request.get("topology_refresh") is not None:
            if request["topology_refresh"]["student_bundle_id"] != parent.record_id:
                raise ValueError(
                    "dynamic task topologies differ from the shared parent"
                )
            atomic_json(context / "topology_refresh.json", request["topology_refresh"])
        atomic_json(context / "evidence.json", evidence)
        atomic_json(context / "feedback_focus.json", focus)
        atomic_json(
            context / "input.json",
            {
                "kind": "shared_distillation",
                "size_reference": "context/size_reference"
                if request.get("size_reference") is not None
                else None,
                "alignment_seconds": request["alignment_seconds"],
                "source_graphs": "context/source_graphs.json",
                **(
                    {"topology_refresh": "context/topology_refresh.json"}
                    if request.get("topology_refresh") is not None
                    else {}
                ),
                "shared_topology": "context/shared_topology.json"
                if topology is not None
                else None,
                "mechanisms": request.get("mechanisms", []),
                "conditional_anchors": request.get("conditional_anchors", {}),
                **{
                    key: request[key]
                    for key in (
                        "round",
                        "mode",
                        "method",
                        "tasks",
                        "new_tasks",
                        "anchors",
                        "criteria",
                        "conditions",
                        "history",
                    )
                },
                "teachers": "context/teachers",
                "evidence": "context/evidence.json",
                "alignments": "context/alignments.json",
                **(
                    {
                        "gap_index": "context/gap_index.json",
                        "gap_query_tool": "query_student_gap",
                        "teacher_unit_bank": "context/teacher_unit_bank.json",
                        "parent_implementation": "context/parent_implementation.json"
                        if (context / "parent_implementation.json").exists()
                        else None,
                    }
                    if request.get("student_gaps") is not None
                    else {}
                ),
                **(
                    {"fragment_transfer": "context/fragment_transfer.json"}
                    if request.get("fragment_transfer") is not None
                    else {}
                ),
            },
        )
        contract = self.runtime_manifest.authoring_contract
        atomic_json(
            context / "gateway.json",
            {
                "operations": contract.operations,
                "tools": contract.tools,
                "episode_limits": self.runtime_manifest.configuration["limits"][
                    "episode"
                ],
                **(
                    {"native_image_contract": contract.native_image_contract}
                    if contract.native_image_contract is not None
                    else {}
                ),
            },
        )
        atomic_json(context / "bundle.schema.json", AgentBundle.model_json_schema())
        atomic_json(
            context / "runtime.schema.json",
            self.runtime_schema(request["conditions"]["controller_request_unit"]),
        )
        atomic_text(
            workspace / "AGENTS.md",
            self._instructions(request),
        )
        atomic_json(workspace / "prepared.json", {"complete": True})
        final_workspace = directory / "workspace"
        os.replace(workspace, final_workspace)
        if self.source_queries:
            seed_parent_interactions(
                final_workspace,
                parent.record_id,
                (
                    json.loads(path.read_text())
                    for path in sorted(
                        directory.parent.parent.glob("*/author/result.json")
                    )
                ),
            )

        return final_workspace, tuple(readable)

    async def __call__(self, request, directory):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".native.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._verify_process_exit(directory)
            result_path = directory / "result.json"
            if result_path.exists():
                bind_json(directory / "request.json", request)
                bind_json(
                    directory / "policy.json", self.policy.model_dump(mode="json")
                )
                return json.loads(result_path.read_text())
            audit = directory / "audit"
            audit.mkdir(exist_ok=True)
            prepared_at = time.monotonic()
            try:
                workspace, evidence_paths = self.prepare(request, directory)
            finally:
                atomic_json(
                    audit / "preparation_usage" / f"{time.time_ns()}.json",
                    Usage(
                        physical_steps=0,
                        model_calls=0,
                        wall_time_s=time.monotonic() - prepared_at,
                        cost_usd=0,
                    ).model_dump(mode="json"),
                )
                self._update_usage(directory)
            while True:
                correction = self._resume_correction(directory)
                native_prompt = self.prompt
                if correction is not None:
                    feedback = json.loads(
                        Path(correction["complete_report"]["path"]).read_text()
                    )
                    atomic_json(workspace / "context/admission_feedback.json", feedback)
                    guidance = ""
                    if self.source_queries:
                        guidance = (
                            "Preserve teacher-derived mechanisms outside the named failure. "
                            "Repair the existing implicated code or fixtures; do not repeat "
                            "the initial consolidation or regenerate unaffected files. "
                            "Read context/runtime.schema.json when repairing scenarios. "
                            "Preserve real image evidence; synthetic fixtures require valid "
                            "encoded image bytes. Run validate_candidate phase=interaction "
                            "and fix its failures before returning. "
                        )
                        if request.get("shared_topology") is not None:
                            guidance += (
                                "Repair the source constraint manifest and topology proposal "
                                "when implicated by the report. Read "
                                "context/shared_implementation.schema.json: source_constraints "
                                "and interfaces must have EXACTLY the enumerated keys, with "
                                "no additional per-condition keys. After writing the manifest, "
                                "run validate_candidate phase=interaction to check the same "
                                "implementation contract before returning. "
                            )
                    native_prompt = self._correction_prompt(
                        {
                            "complete_report": {
                                "path": "context/admission_feedback.json"
                            },
                            "error_type": feedback["error_type"],
                        },
                        guidance=guidance,
                    )
                if not (directory / "native_result.json").exists():
                    query_path = (
                        audit
                        / "queries"
                        / f"{len(list((audit / 'queries').glob('*.json'))):06d}.json"
                    )
                    started = time.monotonic()

                    def begin():
                        atomic_json(
                            query_path,
                            Usage(
                                physical_steps=0,
                                model_calls=None,
                                wall_time_s=0,
                                wall_time_complete=False,
                                cost_usd=None,
                            ).model_dump(mode="json"),
                        )

                    def record(event):
                        with (audit / "events.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(serialize_native_event(event)) + "\n"
                            )

                    system = tuple(
                        Path(p).resolve()
                        for p in (
                            "/usr",
                            "/bin",
                            "/lib",
                            "/lib64",
                            "/etc",
                            "/System",
                            "/dev",
                        )
                        if Path(p).exists()
                    )
                    tools = None
                    transport_failure = None
                    try:
                        if self.source_queries:
                            from reweave.evolution.infrastructure.async_work import (
                                run_blocking,
                            )

                            tools = SharedAuthorTools(
                                workspace,
                                audit / "source_tools.jsonl",
                                validator=lambda **options: self.validate_code(
                                    request, workspace, **options
                                ),
                            )
                            await run_blocking(tools.server.start)
                            from reweave.evolution.authoring.client import (
                                call_development_tool,
                            )

                            await run_blocking(
                                call_development_tool,
                                tools.server.url,
                                "validate_candidate",
                                {"phase": "environment"},
                            )
                        async with offline_admission(directory, required=True):
                            await self._codex(
                                workspace,
                                directory,
                                (
                                    *system,
                                    Path(sys.prefix).resolve(),
                                    Path(sys.base_prefix).resolve(),
                                    *evidence_paths,
                                ),
                                (workspace / "candidate",),
                                record,
                                started,
                                query_path,
                                begin,
                                self._instructions(request),
                                native_prompt,
                                tools.server.url if tools is not None else None,
                            )
                    except Exception as error:
                        if (
                            query_path.exists()
                            and not (directory / "native_result.json").exists()
                            and is_retryable_native_error(error)
                        ):
                            transport_failure = NativeTransportFailure(directory, error)
                        else:
                            raise
                    finally:
                        if tools is not None:
                            from reweave.evolution.infrastructure.async_work import (
                                run_blocking,
                            )

                            await run_blocking(tools.server.stop)
                        if query_path.exists():
                            usage = Usage.model_validate_json(
                                query_path.read_text()
                            ).model_copy(
                                update={
                                    "wall_time_s": time.monotonic() - started,
                                    "wall_time_complete": True,
                                }
                            )
                            atomic_json(query_path, usage.model_dump(mode="json"))
                            self._update_usage(directory)
                    if transport_failure is not None:
                        self._verify_process_exit(directory)
                        await self._wait_for_transport_retry(
                            directory, transport_failure
                        )
                        continue
                self._verify_process_exit(directory)
                try:
                    payload = await self.admit(request, workspace, audit)
                except AUTHOR_VALIDATION_ERRORS as error:
                    if not (directory / "native_result.json").exists():
                        raise
                    check = audit / "checks" / f"shared-rejected-{time.time_ns()}.json"
                    cases_path = audit / "admission.json"
                    cases = (
                        json.loads(cases_path.read_text())
                        if cases_path.exists()
                        else []
                    )
                    self._save_correction(directory, check, error, case_results=cases)
                    from reweave.evolution.validation.admission import (
                        _read_workspace_file,
                    )

                    archived = directory / "audit/corrections" / check.stem
                    for name in (
                        "shared_implementation.json",
                        "topology_proposal.json",
                    ):
                        path = workspace / ".scratch" / name
                        if path.is_file() and not path.is_symlink():
                            try:
                                raw = _read_workspace_file(
                                    workspace, ".scratch/" + name
                                )
                            except OSError:
                                continue
                            (archived / name).write_bytes(raw)
                    continue
                break
            result = {
                **payload,
                "usage": self._update_usage(directory).model_dump(mode="json"),
            }
            bind_json(result_path, result)
            return result

    async def validate_code(self, request, workspace, *, interactions=False):
        """Preview production checks in the author turn; final admission repeats them."""
        cases = []
        before = output_inventory(workspace / "candidate")
        try:
            parent = AgentBundle.model_validate(request["parent"])
            candidate = capture_bundle(workspace / "candidate", parent.runtime).bundle
            contract = json.loads((workspace / "context/gateway.json").read_text())
            await check_runtime_contracts(candidate, contract, case_results=cases)
            component_cases = len(cases)
            if interactions:
                await self._check_interactions(request, workspace, candidate, cases)
                manifest = workspace / ".scratch/shared_implementation.json"
                if request.get("shared_topology") is not None and manifest.exists():
                    validate_implementation(
                        json.loads(
                            _read_workspace_file(
                                workspace, ".scratch/shared_implementation.json"
                            )
                        ),
                        SharedTopology.model_validate(request["shared_topology"]),
                        candidate,
                    )
            if before != output_inventory(workspace / "candidate"):
                raise ValueError("candidate changed during validation")
        except AUTHOR_VALIDATION_ERRORS as error:
            atomic_json(
                workspace / "context/candidate_validation.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "contract_case_results": cases,
                },
            )
            raise ValueError(
                f"{error}; read context/candidate_validation.json"
            ) from error
        atomic_json(
            workspace / "context/candidate_validation.json",
            {"contract_case_results": cases},
        )
        return {
            "phase": "interaction" if interactions else "code",
            "structurally_valid": True,
            "contract_cases": component_cases,
            "interaction_cases": len(cases) - component_cases,
            "physical_execution_tested": False,
            "interaction_tested": interactions,
            "complete_report": "context/candidate_validation.json",
        }

    async def admit(self, request, workspace, audit):
        parent = AgentBundle.model_validate(request["parent"])
        candidate = capture_bundle(workspace / "candidate", parent.runtime).bundle
        cases = []
        admitted_at = time.monotonic()
        contract = json.loads((workspace / "context/gateway.json").read_text())
        try:
            await check_runtime_contracts(candidate, contract, case_results=cases)
            await self._check_interactions(request, workspace, candidate, cases)
        finally:
            atomic_json(audit / "admission.json", cases)
            atomic_json(
                audit / "checks" / f"{time.time_ns()}.json",
                {
                    "wall_time_s": time.monotonic() - admitted_at,
                    "wall_time_complete": True,
                },
            )
            self._update_usage(workspace.parent)
        result = {"bundle": candidate.model_dump(mode="json")}
        if (workspace / SUITE).is_file():
            result["interaction_suite"] = artifact_record(workspace / SUITE)
        topology_started = time.monotonic()
        try:
            if request.get("shared_topology") is not None:
                from reweave.evolution.validation.admission import _read_workspace_file

                topology = SharedTopology.model_validate(request["shared_topology"])
                graphs = {
                    name: SemanticGraph.model_validate(value)
                    for name, value in request["source_graphs"].items()
                }
                proposal_path = workspace / ".scratch/topology_proposal.json"
                if proposal_path.exists() or proposal_path.is_symlink():
                    raw = json.loads(
                        _read_workspace_file(
                            workspace, ".scratch/topology_proposal.json"
                        )
                    )
                    topology = apply_proposal(topology, graphs, raw)
                validate_topology(topology, graphs)
                raw = json.loads(
                    _read_workspace_file(
                        workspace, ".scratch/shared_implementation.json"
                    )
                )
                implementation = validate_implementation(raw, topology, candidate)
                result.update(
                    topology=topology.model_dump(mode="json"),
                    implementation=implementation.model_dump(mode="json"),
                )
        finally:
            atomic_json(
                audit / "checks" / f"topology-{time.time_ns()}.json",
                {
                    "wall_time_s": time.monotonic() - topology_started,
                    "wall_time_complete": True,
                },
            )
        return result

    async def _check_interactions(self, request, workspace, candidate, cases):
        await check_interaction_suite(
            workspace,
            candidate,
            scenario_path=".scratch/tests/interaction.json",
            references=(
                AgentBundle.model_validate(request["parent"]),
                *(AgentBundle.model_validate(t["bundle"]) for t in request["teachers"]),
            ),
            execution=EvaluationConditions.model_validate(request["conditions"]),
            case_results=cases,
        )
