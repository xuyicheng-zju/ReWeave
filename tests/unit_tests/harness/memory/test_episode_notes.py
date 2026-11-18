# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""The real online Planner writes evidence drafts for the existing offline author."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.evidence.records import archive_episode
from reweave.evolution.evidence.trace import (
    build_episode_context,
    resolve_trace_reference,
)
from reweave.memory import ReadOnlyCorpus
from reweave.memory.notes import EpisodeNotes, planner_memory_guidance
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.kernel import EpisodeLimits
from reweave.session import EnvState
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.tools.test_toolkit_contracts import _ContractToolkit


class NoteRobot(_ContractToolkit):
    def __init__(self, output, corpus, *, filesystem_tools_enabled=True):
        from reweave.utils.logging import init_output_dir

        init_output_dir(output)
        super().__init__(output, memory=ReadOnlyCorpus(corpus))
        if not filesystem_tools_enabled:
            self._tools.clear()
            self._register_common_tools(filesystem_tools_enabled=False)
        self.terminal = False
        self.add_tool(
            "move",
            {
                "name": "move",
                "description": "Synthetic physical step",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.move,
        )

    def move(self):
        self.terminal = True
        return {"moved": True}

    def episode_ended(self):
        return self.terminal


@pytest.mark.parametrize("filesystem_tools_enabled", [True, False])
@pytest.mark.parametrize("scope", ["task-family"])
def test_production_planner_writes_reads_revises_then_acts_and_notes_reach_author_navigation(
    tmp_path, filesystem_tools_enabled, scope
):
    corpus = tmp_path / "published"
    (corpus / "global").mkdir(parents=True)
    published = corpus / "global" / "original.md"
    published.write_text("Published fixed knowledge")
    output = tmp_path / "episode"
    robot = NoteRobot(output, corpus, filesystem_tools_enabled=filesystem_tools_enabled)
    note = robot.episode_notes.drafts / scope / "conditions.md"
    seen = []

    def model(messages, info):
        seen.append(info)
        assert "Write useful experience online" in info.instructions
        assert str(robot.episode_notes.drafts) in info.instructions
        names = {tool.name for tool in info.function_tools}
        assert {"write_text_file", "read_text_file", "move"} <= names
        if len(seen) == 1:
            assert robot.state.latest_step is None
            name, args = (
                "write_text_file",
                {
                    "path": str(note),
                    "content": "Hypothesis: uncertain position; inspect fresh evidence.",
                },
            )
        elif len(seen) == 2:
            assert robot.state.latest_step is None
            name, args = "read_text_file", {"path": str(note)}
        elif len(seen) == 3:
            assert "uncertain position" in str(messages)
            assert robot.state.latest_step is None
            name, args = (
                "write_text_file",
                {
                    "path": str(note),
                    "content": "Revised hypothesis: cause remains unknown; prior claim may be wrong.",
                },
            )
        else:
            assert len(seen) == 4
            assert robot.state.latest_step is None
            name, args = "move", {}
        return ModelResponse(
            parts=[ToolCallPart(name, args, tool_call_id=f"call-{len(seen)}")]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="Use actual task evidence.",
        user_message="Synthetic test",
        toolkit=robot,
        max_turns=6,
    )
    assert result.error is None, result.error
    assert len(robot.capture_calls) == 1
    assert note.read_text().startswith("Revised hypothesis")
    assert published.read_text() == "Published fixed knowledge"
    originals = [
        json.loads(path.read_text())
        for path in (robot.episode_notes.root / "revisions").glob("*.json")
    ]
    assert len(originals) == 2
    assert {item["text"] for item in originals} == {
        "Hypothesis: uncertain position; inspect fresh evidence.",
        note.read_text(),
    }
    for record in originals:
        assert record["verification"] == "unverified"
        for source in record["public_sources"]:
            prefix = (output / source["path"]).read_bytes()[: source["prefix_bytes"]]
            assert len(prefix) == source["prefix_bytes"]
    assert all(item["public_sources"] for item in originals)

    robot.export_harness_context(str(output))
    (output / "execution_trace.jsonl").write_text("")
    (output / "transcript_test.json").write_text(
        json.dumps(
            {
                "messages": result.messages,
                "official_success": False,
                "termination_class": "task_failure",
                "task_message": "Synthetic test",
            }
        )
    )
    archive_root = tmp_path / "evidence/.archive/episodes"
    evidence = archive_episode(output, archive_root)
    archived = archive_root / evidence.evidence_id
    reference = build_episode_context(
        tmp_path / "evidence", evidence, archived, include_visuals=False
    )
    index = json.loads((tmp_path / "evidence" / reference["path"]).read_text())
    memory = index["planner_memory"]
    assert len(memory["drafts"]) == 1 and len(memory["revisions"]) == 2
    recovered = [
        resolve_trace_reference(tmp_path / "evidence", ref)
        for ref in memory["revisions"]
    ]
    assert {item["text"] for item in recovered} == {item["text"] for item in originals}
    assert baseline().context.skills == ()  # No automatic promotion into a new policy.


def test_episode_drafts_never_share_mutable_state(tmp_path):
    corpus = tmp_path / "published"
    first = NoteRobot(tmp_path / "first", corpus)
    second = NoteRobot(tmp_path / "second", corpus)
    note = first.episode_notes.drafts / "global" / "lesson.md"
    first.execute_tool(
        "write_text_file", {"path": str(note), "content": "first episode"}
    )
    denied = second.execute_tool("read_text_file", {"path": str(note)})
    assert "another episode" in denied.result["error"]
    alias = tmp_path / "alias.md"
    alias.symlink_to(note)
    assert (
        "another episode"
        in second.execute_tool("read_text_file", {"path": str(alias)}).result["error"]
    )
    # The frozen runtime additionally confines reads to its own episode.
    runtime = BundleRuntime(
        baseline(),
        second,
        "task",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    denied = asyncio.run(runtime._dispatch_tool("read_text_file", {"path": str(note)}))
    assert denied["error_type"] == "PermissionError"
    assert not second.episode_notes.root.exists()
    assert first.episode_notes.root != second.episode_notes.root
    assert "Nothing is automatically" in planner_memory_guidance(first)


@pytest.mark.parametrize(
    "path",
    [
        "../published.md",
        "revisions/forged.json",
        "drafts/global/policy.py",
        "drafts/lesson.md",
        "drafts/unknown/lesson.md",
        "drafts/global/lesson.md",
        "drafts/task-specific/lesson.md",
    ],
)
def test_notes_reject_traversal_archive_edit_and_executable_files(tmp_path, path):
    notes = EpisodeNotes(tmp_path / "episode", EnvState(tmp_path / "episode"))
    with pytest.raises((ValueError, PermissionError)):
        notes.write_text_file(path, "attempted content")
    assert not notes.root.exists()


@pytest.mark.parametrize("scope", ["task-family"])
def test_scoped_drafts_keep_applicability_in_original_path_without_publishing(
    tmp_path, scope
):
    notes = EpisodeNotes(tmp_path / "episode", EnvState(tmp_path / "episode"))
    result = notes.write_text_file(
        f"drafts/{scope}/lesson.md", "Unverified conditional lesson"
    )
    original = json.loads((tmp_path / "episode" / result["revision"]).read_text())
    assert original["draft"] == f"drafts/{scope}/lesson.md"
    assert original["verification"] == "unverified"
    assert notes.read_text_file(original["draft"])["content"] == original["text"]
    assert not (tmp_path / "episode/context/skills").exists()


def test_frozen_runtime_denies_writes_to_published_or_foreign_notes_and_keeps_observation(
    tmp_path,
):
    robot = NoteRobot(tmp_path / "episode", tmp_path / "published")
    runtime = BundleRuntime(
        baseline(),
        robot,
        "task",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    runtime.observation = {"settled": "physical evidence"}
    good = robot.episode_notes.drafts / "task-family" / "lesson.md"
    written = asyncio.run(
        runtime._dispatch_tool(
            "write_text_file", {"path": str(good), "content": "only observed claims"}
        )
    )
    assert written["written"] is True
    assert runtime.observation == {"settled": "physical evidence"}
    for bad in (
        robot.memory.root / "global/x.md",
        tmp_path / "other/planner_memory/drafts/x.md",
        tmp_path / "robot.py",
    ):
        result = asyncio.run(
            runtime._dispatch_tool(
                "write_text_file", {"path": str(bad), "content": "forbidden"}
            )
        )
        assert result["error_type"] == "PermissionError"
        assert not bad.exists()
    assert runtime.kernel._operations["write_text_file"]["effect"] == "write"


@pytest.mark.parametrize("operation", ["read_text_file", "write_text_file", "list_dir"])
def test_frozen_and_fixed_note_tools_reject_symlink_scope_aliases(tmp_path, operation):
    robot = NoteRobot(tmp_path / "episode", tmp_path / "published")
    notes = robot.episode_notes
    target = notes.drafts / "task-family" / "lesson.md"
    notes.write_text_file(str(target), "Original unverified task knowledge")
    alias = notes.drafts / "global" / "alias"
    alias.parent.mkdir()
    alias.symlink_to(target.parent, target_is_directory=True)
    path = alias if operation == "list_dir" else alias / target.name
    arguments = {"path": str(path)}
    if operation == "write_text_file":
        arguments["content"] = "Replacement through another scope"
    direct = robot.execute_tool(operation, arguments).result
    assert "symlinks" in direct["error"]

    runtime = BundleRuntime(
        baseline(),
        robot,
        "task",
        episode_limits=EpisodeLimits(
            program_calls=10, gateway_calls=10, wall_time_s=10
        ),
    )
    runtime.observation = {"settled": "physical evidence"}
    frozen = asyncio.run(runtime._dispatch_tool(operation, arguments))
    assert frozen["error_type"] == "PermissionError"
    assert "symlinks" in frozen["error"]
    assert target.read_text() == "Original unverified task knowledge"
    assert len(list((notes.root / "revisions").glob("*.json"))) == 1
    assert runtime.observation == {"settled": "physical evidence"}
    assert robot.capture_calls == []


def test_next_batch_reference_inventory_exposes_settled_notes_only(tmp_path):
    from reweave.evolution.authoring.references import AuthorReferences
    from reweave.evolution.evidence.records import (
        EpisodeEvidence,
        inventory_episode,
    )

    live_roots = []
    inventories = {}
    for name in ("first", "second"):
        state = EnvState(tmp_path / name)
        notes = EpisodeNotes(state.output_dir, state)
        result = notes.write_text_file(
            str(notes.drafts / "task-family" / "lesson.md"), f"{name} uncertain claim"
        )
        files = inventory_episode(state.output_dir)
        evidence = EpisodeEvidence(evidence_id=name, files=files)
        live_roots.append((evidence, state.output_dir, result["revision"]))

    current = AuthorReferences.create(
        tmp_path / "current/workspace",
        registry=tmp_path / "registry.json",
        evidence_roots={},
    )
    for evidence, root, revision in live_roots:
        ref = {
            "path": f".archive/episodes/{evidence.evidence_id}/{revision}",
            **evidence.files[revision],
            "pointer": "/text",
        }
        with pytest.raises(ValueError, match="outside this request"):
            current.encode(ref)
        # Once the round settles, its exact immutable episode inventory is selected
        # for the next batch. It is never discovered by scanning live siblings.
        destination = tmp_path / ".archive/episodes" / evidence.evidence_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        root.rename(destination)
        inventories[evidence.evidence_id] = destination

    next_batch = AuthorReferences.create(
        tmp_path / "next/workspace",
        registry=tmp_path / "registry.json",
        evidence_roots=inventories,
    )
    for evidence, _, revision in live_roots:
        ref = {
            "path": f".archive/episodes/{evidence.evidence_id}/{revision}",
            **evidence.files[revision],
            "pointer": "/text",
        }
        handle = next_batch.encode(ref)
        original = next_batch.original(handle)
        claim = json.loads(Path(original.path).read_text())
        assert claim["verification"] == "unverified"
        assert claim["text"] in {"first uncertain claim", "second uncertain claim"}
        with pytest.raises(ValueError, match="outside this request"):
            current.encode(ref)


def test_existing_round_barrier_refuses_history_publication_with_live_sibling(tmp_path):
    from reweave.evolution.contracts import EvolutionIncomplete
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
    from tests.unit_tests.harness.evolution.fixtures import protocol

    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    search = GraphEvolution(workspace, None, None)
    state = {"versions": {}, "attempts": {"live_sibling": {"status": "running"}}}
    with pytest.raises(EvolutionIncomplete, match="unresolved prior work"):
        asyncio.run(search._freeze_history(state, {}, 1))
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("scope", ["task-family"])
def test_none_profile_keeps_private_notes_without_enabling_published_or_arbitrary_files(
    tmp_path,
    scope,
):
    from reweave.tools.toolkit import Toolkit

    root = tmp_path / "published"
    leaf = root / "global/existing.md"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("external corpus")
    unrelated = tmp_path / "unrelated.md"
    unrelated.write_text("unrelated file")
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=EnvState(tmp_path / "episode"),
        memory=ReadOnlyCorpus(root, enabled=False),
        filesystem_tools_enabled=False,
    )
    note = toolkit.episode_notes.drafts / scope / "lesson.md"
    assert toolkit.execute_tool(
        "write_text_file", {"path": str(note), "content": "uncertain observation"}
    ).result["written"]
    assert (
        toolkit.execute_tool("read_text_file", {"path": str(note)}).result["content"]
        == "uncertain observation"
    )
    disabled = toolkit.execute_tool("read_text_file", {"path": str(leaf)})
    denied = toolkit.execute_tool("read_text_file", {"path": str(unrelated)})
    assert "none profile" in disabled.result["error"]
    assert "confined" in denied.result["error"]
    assert leaf.read_text() == "external corpus"


def test_frozen_note_tool_contract_is_independent_of_episode_output_paths(tmp_path):
    first = NoteRobot(
        tmp_path / "first", tmp_path / "published", filesystem_tools_enabled=False
    )
    second = NoteRobot(
        tmp_path / "second", tmp_path / "published", filesystem_tools_enabled=False
    )
    assert first.get_tools_contract_spec() == second.get_tools_contract_spec()
    assert str(tmp_path / "first") not in json.dumps(first.get_tools_contract_spec())
    assert "{{output_dir}}/planner_memory/drafts" in json.dumps(
        first.get_tools_contract_spec()
    )
