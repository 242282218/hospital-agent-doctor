"""Python 3.9 container compatibility gate for the runtime package.

The ModelScope studio image pins python:3.9 while local dev runs 3.11+,
so syntax/runtime-annotation drift would only explode at deploy time.
Run: python scripts/check_py39_compat.py [package_dir ...]
Exit code 0 = compatible, 1 = violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Runtime APIs that do not exist on 3.9; annotations are handled separately.
BANNED_CALL_ATTRS = {"pairwise", "anext", "aiter", "bit_count"}
BANNED_DATACLASS_KWARGS = {"slots", "kw_only", "match_args"}


def _has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def _annotation_nodes(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (
                *args.posonlyargs, *args.args, *args.kwonlyargs,
                *filter(None, (args.vararg, args.kwarg)),
            ):
                if arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _contains_pep604_union(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)
        for sub in ast.walk(node)
    )


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    problems: list[str] = []
    try:
        # feature_version rejects 3.10+ syntax such as match statements.
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: 3.9-incompatible syntax: {exc.msg}"]

    future_ok = _has_future_annotations(tree)
    if not future_ok:
        for annotation in _annotation_nodes(tree):
            if _contains_pep604_union(annotation):
                problems.append(
                    f"{path}:{annotation.lineno}: PEP 604 union evaluated at "
                    "runtime without `from __future__ import annotations`"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in BANNED_CALL_ATTRS:
                problems.append(
                    f"{path}:{node.lineno}: `{name}` requires Python 3.10+"
                )
            if name == "zip" and any(kw.arg == "strict" for kw in node.keywords):
                problems.append(
                    f"{path}:{node.lineno}: zip(strict=...) requires 3.10+"
                )
            if name == "dataclass":
                for kw in node.keywords:
                    if kw.arg in BANNED_DATACLASS_KWARGS:
                        problems.append(
                            f"{path}:{node.lineno}: dataclass({kw.arg}=...) "
                            "requires 3.10+"
                        )
    # `Alias = A | B` at module scope with bare type-like operands evaluates
    # the union eagerly on import; runtime set unions inside functions are
    # legitimate 3.9 code, so only top-level statements are inspected.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
            value = node.value
            if isinstance(value.op, ast.BitOr) and all(
                isinstance(side, (ast.Name, ast.Attribute, ast.Subscript))
                or (isinstance(side, ast.Constant) and side.value is None)
                for side in (value.left, value.right)
            ):
                problems.append(
                    f"{path}:{node.lineno}: possible runtime PEP 604 union in "
                    "module-level assignment (verify manually)"
                )
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(p) for p in argv[1:]] or [Path("agent")]
    all_problems: list[str] = []
    file_count = 0
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            file_count += 1
            all_problems.extend(check_file(path))
    print(f"checked {file_count} files under: {', '.join(map(str, roots))}")
    if all_problems:
        print(f"FAIL: {len(all_problems)} Python 3.9 compatibility problem(s)")
        for problem in all_problems:
            print(f"  {problem}")
        return 1
    print("PASS: no Python 3.9 compatibility problems detected")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
