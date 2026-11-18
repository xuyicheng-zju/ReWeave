# Evolution implementation

The package is organized by responsibility. `bundle.py`, `contracts.py`, and
`runtime_manifest.py` remain at the root because both online execution and offline
search use these shared data contracts.

| Directory | Responsibility | Main entry points |
| --- | --- | --- |
| `search/` | Finite task search, development evaluation, selection reports and ablation controls | `engine.py`, `campaign.py`, `rollout.py` |
| `graph/` | Topology analysis, validation, projection, matching, retrieval and width/parent selection | `analysis.py`, `projection.py`, `selection.py` |
| `authoring/` | Native candidate author, prompts, workspace boundary and development tools | `author.py`, `candidate.py`, `tools.py` |
| `validation/` | Candidate admission, recorded-context replay and synthetic full-runtime checks | `admission.py`, `context_replay.py`, `interaction.py` |
| `distillation/` | Source-only shared harness construction, topology factoring and transfer assessment | `engine.py`, `author.py`, `stream.py` |
| `evidence/` | Immutable episode records, trajectory views, media retention and remote evidence access | `records.py`, `trajectory.py`, `trace.py` |
| `infrastructure/` | Native sessions, accounting, process ownership, slots and SSH transport | `native_codex.py`, `session.py`, `campaign_slots.py` |

Import from the owning module, for example
`from reweave.evolution.search.engine import GraphEvolution`. Each entry point imports the subsystem that owns its contract. Synthetic
interaction checks exercise the production runtime and record scripted inputs.
