# Installation

ReWeave supports Python 3.11 and 3.12 on Linux and macOS. The CPU smoke test
uses scripted responses. Use a Linux GPU environment for simulator execution.
Linux native authoring requires Landlock and libseccomp. On Ubuntu, install
`libseccomp2` for the runtime. macOS native authoring uses Seatbelt.

## Core package

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
reweave --help
reweave-evolve --help
reweave-distill --help
```

Install `.[test,graspgen]` for the development test suite, `.[retrieval]` for
local graph embeddings, or `.[remote]` for SSH-backed sensor archives.

## Language models

Install the [Codex CLI](https://developers.openai.com/codex/cli) and run
`codex login` for the default local account mode. A session-specific copy of
its login file can be stored in the local run directory. Keep that directory
private. Use `--model` to choose an available online model and
[an author policy](../examples/graph_evolution/author_policy.json) to set the
offline model.

For a Responses API provider, export `OPENAI_API_KEY` and use
`--codex-auth-mode api` online or `"codex_auth_mode": "api"` in the Codex
author policy. `REWEAVE_CODEX_API_BASE_URL` supplies an optional compatible base
URL. The default is the public OpenAI Responses API.

The Claude backend reads `ANTHROPIC_API_KEY` from the environment.
An optional `ANTHROPIC_BASE_URL` selects a compatible endpoint. The Cursor
backend uses a logged-in native Cursor Agent installation, with a private
profile under `~/.config/reweave/cursor`. Its CLI details and model availability
depend on the installed Cursor version.

## Simulator environments

Install the benchmark, its policy runtime, and ReWeave in compatible
environments. Follow upstream instructions for simulator and model assets.

| Backend | Requirements | Local configuration |
| --- | --- | --- |
| LIBERO-PRO | LIBERO-PRO, RLinf's LIBERO adapter, MuJoCo, a compatible OpenPI Pi0.5 policy and SAM3 | `--libero-type pro`, `PI05_CHECKPOINT_PATH`, `SAM3_CHECKPOINT_PATH` |
| RoboDojo | RoboDojo, its RLinf `CollectEnv` bridge, Isaac Sim, XPolicyLab, the matching 14-value Pi0.5 model and SAM3 | `ROBODOJO_ASSETS_ROOT`, `PI05_CHECKPOINT_PATH`, `SAM3_CHECKPOINT_PATH`, `--sim-python`, `--pi05-python` |

The Pi0.5 policy can require `PI05_NORM_STATS_PATH` for separate normalization
statistics. RoboDojo accepts `--source-root` and `--xpolicylab-root` for explicit
checkouts. LIBERO accepts compatible service URLs via `--env-endpoint`,
`--vla-endpoint` and `--sam3-endpoint`. Use the backend-specific CLI help to see
supported resource flags. The [RoboDojo guide](../reweave/benchmarks/robodojo/README.md)
describes asset preparation and the interactive capture interface.

[Upstream components](../third_party/README.md) lists source links and the
simulator build requirements. Record the installed revisions with your runs.
