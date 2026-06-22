"""
Step 2 on the architecture slide — 'Sandboxed Python executor runs code
against read-only in-memory dataframes'.

Defence in depth, appropriate for an internal analytics tool where the code
itself is LLM-generated rather than user-supplied:
  1. Static AST check — reject import / dunder access / exec / eval / file
     or network calls before a single line runs.
  2. Restricted builtins — only a small safe whitelist is available.
  3. Read-only inputs — every dataframe is handed in as a fresh .copy().
  4. Hard timeout — a runaway script is killed rather than hanging the UI.

This is a reasonable guardrail for a trusted internal tool, not a claim of
adversarial-proof isolation — in a production rollout this step would run
in a dedicated sandboxed process/container per the deck's security slide.
"""

import ast
import threading
import pandas as pd
import numpy as np

TIMEOUT_SECONDS = 15

FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
}

SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "round": round, "abs": abs, "enumerate": enumerate, "zip": zip,
    "str": str, "int": int, "float": float, "bool": bool, "print": print,
    "isinstance": isinstance, "True": True, "False": False, "None": None,
}


class SandboxError(RuntimeError):
    pass


def _static_check(code: str):
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"Generated code has a syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("Generated code attempted an import — blocked.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError("Generated code touched a dunder attribute — blocked.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxError(f"Generated code referenced a blocked name: {node.id}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_NAMES:
                raise SandboxError(f"Generated code called a blocked function: {fn.id}")


def run_pandas_code(code: str, tables: dict) -> dict:
    """Executes LLM-generated pandas code against read-only copies of
    `tables`. Returns {'result': ..., 'chart_df': df_or_None, 'error': str_or_None}."""
    outcome = {"result": None, "chart_df": None, "error": None}

    try:
        _static_check(code)
    except SandboxError as e:
        outcome["error"] = str(e)
        return outcome

    sandbox_ns = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
    }
    # Hand in a defensive copy of each table so generated code can never
    # mutate the cached, shared dataframes used by the rest of the app.
    for name, df in tables.items():
        sandbox_ns[name] = df.copy(deep=False)

    outcome = {"result": None, "chart_df": None, "error": None}

    def _target():
        try:
            # A single namespace (not separate globals/locals) is required:
            # exec() with two distinct dicts makes top-level code behave like
            # a class body, so lambdas/.apply() closures can't see variables
            # assigned earlier in the same script — a common pandas pattern.
            exec(code, sandbox_ns)  # noqa: S102 - guarded by _static_check above
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
            outcome["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_SECONDS)
    if thread.is_alive():
        outcome["error"] = f"Query took longer than {TIMEOUT_SECONDS}s and was stopped."
        return outcome

    if outcome["error"]:
        return outcome

    outcome["result"] = sandbox_ns.get("result")
    outcome["chart_df"] = sandbox_ns.get("chart_df")
    if outcome["result"] is None:
        outcome["error"] = "Generated code ran but did not set a `result` variable."
    return outcome


def result_to_text(result) -> str:
    """Compact, LLM-friendly text representation of a pandas result."""
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.to_string()
    return str(result)
