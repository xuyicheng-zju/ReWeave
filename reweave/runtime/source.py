# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Static validation for coding-agent-authored Python packages."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from reweave.evolution.bundle import PythonProgram

# Ordinary local computation is unrestricted; only modules with host I/O are absent.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "asyncio",
        "collections",
        "dataclasses",
        "decimal",
        "enum",
        "functools",
        "itertools",
        "json",
        "math",
        "numpy",
        "operator",
        "re",
        "statistics",
        "typing",
    }
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
)


@dataclass(frozen=True, slots=True)
class SourceValidation:
    """Static facts derived from an authored source tree."""

    imported_roots: tuple[str, ...]
    local_modules: tuple[str, ...]
    runtime_calls: tuple["StaticRuntimeCall", ...]
    run_local_callees: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaticRuntimeCall:
    """A generated runtime call whose operation name is statically visible."""

    method: str
    operation: str
    argument_keys: tuple[str, ...] | None
    literal_arguments: dict[str, Any] | None
    function: str
    in_loop: bool


def _module_names(program: PythonProgram) -> set[str]:
    modules: set[str] = set()
    for code_file in program.files:
        path = code_file.path
        if path.endswith("/__init__.py"):
            module = path.removesuffix("/__init__.py").replace("/", ".")
        elif path == "__init__.py":
            module = ""
        else:
            module = path.removesuffix(".py").replace("/", ".")
        if module:
            modules.add(module)
            modules.add(module.split(".", 1)[0])
    return modules


def validate_program_source(
    program: PythonProgram,
    *,
    extra_entrypoints: tuple[str, ...] = (),
) -> SourceValidation:
    """Compile the full tree and inspect imports; OS isolation is still mandatory."""
    local_modules = _module_names(program)
    imported_roots: set[str] = set()
    runtime_calls: list[StaticRuntimeCall] = []
    run_local_callees: list[str] = []
    required_entrypoints = {program.entrypoint, *extra_entrypoints}
    found_entrypoints: set[str] = set()

    for code_file in program.files:
        try:
            tree = ast.parse(code_file.source, filename=code_file.path, mode="exec")
            compile(tree, code_file.path, "exec")
        except SyntaxError as exc:
            raise ValueError(
                f"generated file {code_file.path!r} is not valid Python: {exc}"
            ) from exc

        file_module = (
            code_file.path.removesuffix("/__init__.py").replace("/", ".")
            if code_file.path.endswith("/__init__.py")
            else code_file.path.removesuffix(".py").replace("/", ".")
        )
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
                imported_roots.update(roots)
                unknown = set(roots) - ALLOWED_IMPORT_ROOTS - local_modules
                if unknown:
                    raise ValueError(
                        f"generated file {code_file.path!r} imports forbidden modules: "
                        f"{sorted(unknown)}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                root = (node.module or "").split(".", 1)[0]
                if root:
                    imported_roots.add(root)
                if root not in ALLOWED_IMPORT_ROOTS and root not in local_modules:
                    raise ValueError(
                        f"generated file {code_file.path!r} imports forbidden module "
                        f"{node.module!r}"
                    )
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _FORBIDDEN_CALL_NAMES
                ):
                    raise ValueError(
                        f"generated file {code_file.path!r} calls forbidden builtin "
                        f"{node.func.id!r}"
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "runtime"
                    and node.func.attr in {"observe", "act", "control", "plan"}
                ):
                    method = node.func.attr
                    operation_node = (
                        ast.Constant(value="model:plan")
                        if method == "plan"
                        else (node.args[0] if node.args else ast.Constant(value=""))
                    )
                    if not (
                        isinstance(operation_node, ast.Constant)
                        and isinstance(operation_node.value, str)
                        and operation_node.value
                    ):
                        continue
                    arguments_node = (
                        None if method == "plan" or len(node.args) < 2 else node.args[1]
                    )
                    argument_keys: tuple[str, ...] | None = None
                    literal_arguments: dict[str, Any] | None = None
                    if isinstance(arguments_node, ast.Dict):
                        keys: list[str] = []
                        for key in arguments_node.keys:
                            if not (
                                isinstance(key, ast.Constant)
                                and isinstance(key.value, str)
                            ):
                                keys = []
                                break
                            keys.append(key.value)
                        if keys or not arguments_node.keys:
                            argument_keys = tuple(keys)
                        try:
                            literal = ast.literal_eval(arguments_node)
                        except (ValueError, TypeError):
                            literal = None
                        if isinstance(literal, dict):
                            literal_arguments = literal
                    ancestor = parents.get(node)
                    function = f"{code_file.path}:<module>"
                    in_loop = False
                    while ancestor is not None:
                        if isinstance(
                            ancestor,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
                        ):
                            function = f"{code_file.path}:{getattr(ancestor, 'name', '<lambda>')}:{ancestor.lineno}"
                            break
                        in_loop |= isinstance(
                            ancestor,
                            (
                                ast.For,
                                ast.AsyncFor,
                                ast.While,
                                ast.comprehension,
                                ast.ListComp,
                                ast.SetComp,
                                ast.DictComp,
                                ast.GeneratorExp,
                            ),
                        )
                        ancestor = parents.get(ancestor)
                    runtime_calls.append(
                        StaticRuntimeCall(
                            method="control" if method == "plan" else method,
                            operation=str(operation_node.value),
                            argument_keys=argument_keys,
                            literal_arguments=literal_arguments,
                            function=function,
                            in_loop=in_loop,
                        )
                    )
            elif isinstance(node, ast.Attribute) and node.attr in {
                "__class__",
                "__dict__",
                "__globals__",
                "__subclasses__",
            }:
                raise ValueError(
                    f"generated file {code_file.path!r} accesses forbidden attribute "
                    f"{node.attr!r}"
                )

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if file_module == program.entrypoint.split(":", 1)[0] and "run" in functions:
            run_local_callees = [
                child.func.id
                for child in ast.walk(functions["run"])
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            ]
        found_entrypoints.update(
            entrypoint
            for entrypoint in required_entrypoints
            if entrypoint.split(":", 1)[0] == file_module
            and entrypoint.split(":", 1)[1] in functions
        )

    missing_entrypoints = sorted(required_entrypoints - found_entrypoints)
    if missing_entrypoints:
        raise ValueError(f"entrypoints are not defined: {missing_entrypoints}")
    return SourceValidation(
        imported_roots=tuple(sorted(imported_roots)),
        local_modules=tuple(sorted(local_modules)),
        runtime_calls=tuple(runtime_calls),
        run_local_callees=tuple(run_local_callees),
    )
