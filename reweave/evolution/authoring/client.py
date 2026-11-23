# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Small unprivileged client for workspace-bound Host development tools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def development_client() -> str:
    """Render the workspace convenience wrapper around this client."""
    return '''"""Use the same Host tools from the CLI or a Python analysis script."""
import os
import runpy
from pathlib import Path

_client = runpy.run_path(str(Path(__file__).with_name("host_client.py")))
DevelopmentToolError = _client["DevelopmentToolError"]

def call(name, **arguments):
    return _client["call_development_tool"](os.environ["REWEAVE_AUTHOR_TOOLS_URL"], name, arguments)

if __name__ == "__main__":
    raise SystemExit(_client["main"]())
'''


class DevelopmentToolError(RuntimeError):
    """A failed Host tool call retaining its compact MCP result."""

    def __init__(self, result: dict):
        self.result = result
        super().__init__(json.dumps(result, ensure_ascii=False))


def call_development_tool(url: str, name: str, arguments: dict) -> dict:
    """Call one workspace-bound Host tool and fail on an MCP error result."""
    result = _request_development_tool(url, name, arguments)
    if result.get("isError"):
        raise DevelopmentToolError(result)
    return result["structuredContent"]


def _request_development_tool(url: str, name: str, arguments: dict) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("development tools require the local Host endpoint")
    request = Request(
        url,
        data=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with build_opener(ProxyHandler({})).open(request) as response:
        envelope = json.load(response)
    if "error" in envelope:
        raise RuntimeError(f"development protocol error: {envelope['error']}")
    return envelope["result"]


def _write_report(result: dict, output: Path) -> dict:
    receipt = result.get("structuredContent", {})
    reference = receipt["complete_report"]
    entry = call_development_tool(
        os.environ["REWEAVE_AUTHOR_TOOLS_URL"],
        "read_original",
        {"reference": reference},
    )["value"]
    complete = entry["result"] if entry.get("ok") else entry
    expanded = {
        **result,
        "structuredContent": complete,
        "content": [{"type": "text", "text": json.dumps(complete, ensure_ascii=False)}],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(expanded, ensure_ascii=False, indent=2))
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native author tools; final admission belongs to the Host."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument(
        "--output",
        type=Path,
        help="Save the complete code-check report under .scratch/tests/.",
    )
    commands.add_parser("validate-graph")
    graph = commands.add_parser("graph")
    graph.add_argument(
        "view",
        choices=(
            "summary",
            "features",
            "node",
            "episode",
            "transitions",
            "witness",
            "program",
            "version",
            "comparison",
        ),
    )
    graph.add_argument("--layer", choices=("execution", "semantic", "map"))
    graph.add_argument("--identity")
    graph.add_argument("--first-frame", type=int)
    graph.add_argument("--last-frame", type=int)
    graph.add_argument("--features", nargs="+")
    trace = commands.add_parser("trace")
    trace.add_argument("evidence_id")
    trace.add_argument(
        "view", choices=("executions", "timeline", "telemetry", "diagnostics")
    )
    trace.add_argument("--execution-ref")
    trace.add_argument("--start-step", type=int)
    trace.add_argument("--end-step", type=int)
    trace.add_argument("--features", nargs="+")
    original = commands.add_parser("original")
    original.add_argument(
        "reference",
        help="Issued file handle, optionally followed by # and a JSON pointer",
    )
    source = commands.add_parser("source")
    source.add_argument("source_id")
    source.add_argument("path")
    args = vars(parser.parse_args())
    command = args.pop("command")
    output = args.pop("output", None)
    if command == "validate":
        args["phase"] = "code"
    result = _request_development_tool(
        os.environ["REWEAVE_AUTHOR_TOOLS_URL"],
        {
            "validate": "validate_candidate",
            "validate-graph": "validate_graph",
            "trace": "read_trace",
            "graph": "read_graph",
            "original": "read_original",
            "source": "read_source",
        }[command],
        {key: value for key, value in args.items() if value is not None},
    )
    failed = bool(result.get("isError"))
    if output is not None:
        complete = _write_report(result, output)
        print(
            json.dumps(
                {
                    "report": str(output.resolve()),
                    "is_error": failed,
                    "structurally_valid": complete.get("structurally_valid", False),
                    "error_type": complete.get("error_type"),
                    "cases": [
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
                        for case in complete.get("contract_case_results", [])
                    ],
                },
                ensure_ascii=False,
            )
        )
    elif failed:
        raise DevelopmentToolError(result)
    else:
        print(json.dumps(result["structuredContent"], ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
