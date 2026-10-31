import ast
import json
import keyword
import re
from pathlib import Path
from xml.etree import ElementTree

from .git_service import file_history
from .repository_service import RepositoryStore, run_git
from .schemas import EVIDENCE_CATEGORY_BY_TYPE, EVIDENCE_KIND_BY_TYPE

MAX_EVIDENCE = 64
MAX_EVIDENCE_CHARS = 40_000
MAX_CROSS_REFERENCE_CARDS = 16
MAX_SCAN_FILES = 500
MAX_SCAN_BYTES = 50_000_000
MAX_SYMBOLS = 8
MAX_SELECTED_CODE_CHARS = 16_000
EXCLUDED_SCAN_PARTS = {".git", "node_modules", ".venv", "venv", "dist", "build", "coverage", "__pycache__", ".next", "vendor"}


def collect_evidence(store: RepositoryStore, repo_id: str, file_path: str,
                     start: int, end: int, question: str = "", dependency_depth: int = 2) -> tuple[str, list[dict], dict, dict]:
    repo = store.get(repo_id)
    path = store.resolve_file(repo_id, file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if start > len(lines) or end > len(lines) or end < start or end - start > 250:
        from fastapi import HTTPException
        raise HTTPException(400, f"Select a valid range of up to 250 lines (file has {len(lines)} lines).")
    code = "\n".join(f"{number:>4}: {lines[number - 1]}" for number in range(start, end + 1))
    if len(code) > MAX_SELECTED_CODE_CHARS:
        from fastapi import HTTPException
        raise HTTPException(400, "Selected code is too large for one investigation. Reduce the line range and retry.")
    history = file_history(store, repo_id, file_path, start, end)
    items: list[dict] = []
    evidence_chars = 0
    cross_reference_cards = 0

    def add(kind: str, location: str, excerpt: str, reason: str, *, path: str | None = None,
            commit: str | None = None, line_range: str | None = None, url: str | None = None) -> None:
        nonlocal evidence_chars, cross_reference_cards
        if not excerpt.strip() or len(items) >= MAX_EVIDENCE:
            return
        is_cross_reference = kind in {"assignment", "read", "related_function"}
        if is_cross_reference and cross_reference_cards >= MAX_CROSS_REFERENCE_CARDS:
            return
        max_excerpt = MAX_SELECTED_CODE_CHARS if kind == "source" else 8000 if kind in {"source_function", "related_function"} else 1800
        excerpt = excerpt[:max_excerpt]
        remaining_chars = MAX_EVIDENCE_CHARS - evidence_chars
        if remaining_chars < 100:
            return
        excerpt = excerpt[:remaining_chars]
        evidence_id = f"E{len(items) + 1:02d}"
        category = EVIDENCE_CATEGORY_BY_TYPE[kind]
        items.append({"id": evidence_id, "type": kind, "category": category, "location": location, "path": path,
                      "evidence_kind": EVIDENCE_KIND_BY_TYPE[kind],
                      "commit": commit, "url": url, "line_range": line_range,
                      "excerpt": excerpt, "relevance_reason": reason})
        evidence_chars += len(excerpt)
        if is_cross_reference:
            cross_reference_cards += 1

    add("source", file_path, code, "Current source lines only; this is not evidence of historical intent.",
        path=file_path, line_range=f"{start}-{end}")
    history_ids = {commit["hash"] for commit in history["commits"]}
    line_ids = set(history["line_trace_commits"])
    selected_ids = set(history["selected_line_commits"])
    for commit in history["commits"][:8]:
        changed = ", ".join(entry["status"] + " " + ", ".join(entry["paths"])
                             for entry in commit["changed_paths"][:6]) or "changed paths unavailable"
        commit_path = _path_for_commit(commit, file_path)
        detail = (f"Commit: {commit['hash']}\nParent: {commit['parent'] or 'none (root commit)'}\n"
                  f"Subject: {commit['subject']}\nAuthor: {commit['author']}\nDate: {commit['date']}\n"
                  f"Changed paths: {changed}\nSelected lines attributed here: {commit['hash'] in selected_ids}")
        add("commit", f"commit:{commit['short_hash']}", detail,
            "Git commit metadata. Its message is the author's stated intent, not independent proof.",
            path=commit_path, commit=commit["hash"], url=commit.get("url"))

    for trace in history["line_trace"][:8]:
        if trace["hash"] not in history_ids:
            # A line-range commit can be reached through a rename without being in the current path's log.
            history_ids.add(trace["hash"])
        detail = (f"Commit: {trace['hash']}\nParent: {trace['parent'] or 'none'}\n"
                  f"Subject: {trace['subject']}\nAuthor: {trace['author']}\nDate: {trace['date']}\n"
                  f"Patch for the selected line range:\n{trace['patch'][-1800:]}")
        trace_commit = next((item for item in history["commits"] if item["hash"] == trace["hash"]), None)
        trace_path = _path_for_commit(trace_commit, file_path) if trace_commit else file_path
        add("line_history", f"selected lines · commit:{trace['hash'][:10]}", detail,
            "Git -L traced the evolution of the selected line range.", path=trace_path, commit=trace["hash"])

    # Collapse adjacent lines attributed to the same commit into one evidence card.
    blame_groups: list[list[dict]] = []
    for entry in history["blame"]:
        if (blame_groups and blame_groups[-1][-1]["line"] + 1 == entry["line"]
                and blame_groups[-1][-1]["original_line"] + 1 == entry["original_line"]
                and all(blame_groups[-1][0].get(key) == entry.get(key) for key in ("commit", "path", "author", "date", "summary"))):
            blame_groups[-1].append(entry)
        else:
            blame_groups.append([entry])
    for group in blame_groups:
        first, last = group[0], group[-1]
        details = ", ".join(str(item["original_line"]) for item in group[:20])
        detail = (f"Current lines {first['line']}-{last['line']} originated at {first['path']}:{first['original_line']}"
                  f"{' and following lines' if len(group) > 1 else ''}\nCommit: {first['commit']}\nAuthor: {first['author']}\n"
                  f"Timestamp: {first['date']}\nCommit summary: {first['summary']}\nOriginal line numbers: {details}")
        add("blame", f"{first['path']}:{first['original_line']}-{last['original_line']}", detail,
            "Git blame attributes this contiguous line range; it does not establish why it was written.",
            path=first["path"], commit=first["commit"], line_range=f"{first['original_line']}-{last['original_line']}")

    for diff in history["diffs"]:
        commit = next((item for item in history["commits"] if item["hash"] == diff["commit"]), None)
        diff_paths = [path for entry in diff["changed_paths"] for path in entry["paths"]]
        relevant_path = next((candidate for candidate in diff_paths if candidate == file_path),
                             diff_paths[-1] if diff_paths else file_path)
        is_selected = diff["commit"] in line_ids or diff["commit"] in selected_ids
        add("diff", f"commit:{diff['commit'][:10]} · {relevant_path}",
            f"Parent: {diff['parent'] or 'none'}\nSelected-line history match: {is_selected}\n{diff['excerpt']}",
            "Git before-and-after diff for a real changed path.", path=relevant_path,
            commit=diff["commit"], url=diff.get("url"))

    symbols = _selected_symbols(file_path, lines, start, end, question)
    candidate_files = store.context_files(repo_id)
    candidate_files = _rank_candidates(candidate_files, file_path, symbols, question, dependency_depth, "\n".join(lines),
                                      lambda candidate: store.resolve_evidence_file(repo_id, candidate).read_text(encoding="utf-8", errors="replace"))
    scope = {"files_available": len(candidate_files), "files_inspected": 0,
             "complete": True, "evidence_complete": True, "symbols_traced": symbols,
             "dependency_depth": dependency_depth, "files_skipped": [], "inspected_paths": [],
             "files_skipped_count": 0,
             "category_counts": {"A_source": 0, "B_history": 0, "C_diff": 0, "D_cross_references": 0, "E_tests": 0, "F_documentation": 0, "G_configuration": 0},
             "scope_description": "Git-tracked and non-ignored worktree files with supported source, test, and documentation extensions; excludes generated/vendor folders and files over 250 KB."}
    if len(candidate_files) >= 1500:
        scope["complete"] = False
    scan_bytes = 0
    matches: dict[tuple[str, str, str], int] = {}
    for item in candidate_files:
        candidate = item["path"]
        if any(part in EXCLUDED_SCAN_PARTS for part in Path(candidate).parts):
            _skip(scope, candidate, "generated or vendor directory")
            continue
        if item["size"] > 250_000:
            scope["complete"] = False
            _skip(scope, candidate, "over 250 KB")
            continue
        if scope["files_inspected"] >= MAX_SCAN_FILES or scan_bytes + item["size"] > MAX_SCAN_BYTES:
            scope["complete"] = False
            _skip(scope, candidate, "scan file or byte limit")
            break
        try:
            candidate_path = store.resolve_evidence_file(repo_id, candidate)
            text = candidate_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            _skip(scope, candidate, "could not read safely")
            continue
        scope["files_inspected"] += 1
        scope["inspected_paths"].append(candidate)
        scan_bytes += len(text.encode("utf-8", errors="replace"))
        _add_symbol_context(candidate, text, symbols, add, matches, selected_path=file_path,
                            selected_start=start, selected_end=end)

    # Add related tests and repository docs even when they do not reference a selected symbol.
    for candidate in candidate_files:
        name = Path(candidate["path"]).name.lower()
        is_doc = name.startswith(("readme", "contributing", "changelog"))
        is_dependency_config = name.startswith("requirements") and Path(name).suffix == ".txt"
        if not is_doc and not is_dependency_config and name not in {"pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg"}:
            continue
        if len(items) >= MAX_EVIDENCE:
            scope["evidence_complete"] = False
            break
        try:
            related = store.resolve_evidence_file(repo_id, candidate["path"])
            excerpt = related.read_text(encoding="utf-8", errors="replace")[:1600]
        except Exception:
            continue
        if is_dependency_config:
            kind = "dependency_configuration"
            category = "G_configuration"
            reason = "Dependency manifest read from the current worktree."
        elif name in {"pytest.ini", "tox.ini"} or (name == "pyproject.toml" and re.search(r"(?im)^\s*\[tool\.pytest(?:\.|\])", excerpt)) or (name == "setup.cfg" and re.search(r"(?im)^\s*\[tool:pytest\]", excerpt)):
            kind = "test_configuration"
            category = "G_configuration"
            reason = "Test-tool configuration read from the current worktree."
        elif name in {"pyproject.toml", "setup.cfg"} and re.search(r"(?im)^\s*(?:dependencies\s*=|install_requires\s*=|\[project\])", excerpt):
            kind = "dependency_configuration"
            category = "G_configuration"
            reason = "Project dependency configuration read from the current worktree."
        elif name in {"pyproject.toml", "setup.cfg"}:
            continue
        else:
            kind = "documentation"
            category = "F_documentation"
            reason = "Repository documentation read from the current worktree."
        add(kind, candidate["path"], excerpt, reason,
            path=candidate["path"])

    test_discovery = _discover_test_relationships(repo_id, file_path, "\n".join(lines), start, end, candidate_files, store)
    for relationship in test_discovery["references"]:
        if relationship["classification"] not in {"DIRECT", "INDIRECT"}:
            continue
        reason = relationship["relationship_reason"]
        if len(items) >= MAX_EVIDENCE:
            scope["evidence_complete"] = False
            break
        try:
            text = store.resolve_evidence_file(repo_id, relationship["path"]).read_text(encoding="utf-8", errors="replace")
            excerpt = _test_excerpt(text, relationship["line_ranges"])
        except OSError:
            relationship["classification"] = "UNKNOWN"
            relationship["relationship_reason"] = "Test file could not be read for evidence extraction."
            continue
        evidence_id = f"E{len(items) + 1:02d}"
        add("test", f"{relationship['path']}:{', '.join(relationship['line_ranges'])}", excerpt, reason,
            path=relationship["path"], line_range=", ".join(relationship["line_ranges"]))
        relationship["evidence_references"] = [evidence_id] if len(items) and items[-1]["id"] == evidence_id else []
    test_discovery["total_test_files_found"] = len(test_discovery.pop("test_files"))
    test_discovery["counts"] = {category: sum(item["classification"] == category for item in test_discovery["references"])
                                 for category in ("DIRECT", "INDIRECT", "UNRELATED", "UNKNOWN")}
    coverage = _coverage_measurement(repo_id, candidate_files, store)
    test_discovery.update(coverage)
    test_discovery["coverage_unknown"] = coverage["coverage_status"] != "measured"
    _rank_evidence(items, test_discovery)
    scope["test_discovery"] = test_discovery
    scope["related_tests_found"] = bool(test_discovery["counts"]["DIRECT"] or test_discovery["counts"]["INDIRECT"])
    for item in items:
        scope["category_counts"][item["category"]] += 1
    if len(items) >= MAX_EVIDENCE:
        scope["evidence_complete"] = False
        scope["note"] = "Evidence cards were capped after scanning; not every relevant match could be included."
    if evidence_chars >= MAX_EVIDENCE_CHARS:
        scope["evidence_complete"] = False
        scope["note"] = "Evidence context reached the prompt size limit; not every relevant match could be included."

    if not symbols:
        scope["note"] = "No stable identifiers were extracted from the selected source range."
    elif not any(kind in {"assignment", "read", "related_function", "test"} for _, _, kind in matches):
        scope["note"] = "No matching repository references were found in the inspected scope; this does not prove a symbol is unused."
    if not scope["complete"]:
        scan_note = (f"Repository scan was partial: inspected {scope['files_inspected']} of "
                     f"{scope['files_available']} listed files within the scan limits.")
        scope["note"] = (scope.get("note", "") + " " + scan_note).strip()
    if not scope["evidence_complete"]:
        scope["note"] = (scope.get("note", "") + " Evidence selection was partial due to its size limits.").strip()
    return code, items, history, scope


def _discover_test_relationships(repo_id: str, file_path: str, selected_source: str, start: int, end: int,
                                candidates: list[dict], store: RepositoryStore) -> dict:
    """Classify each discovered test reference once; category counts derive from this canonical list."""
    selected_names, _ = _selected_function_relationships(selected_source, start, end)
    related_names = _repository_related_symbols(repo_id, file_path, selected_source, start, end, candidates, store, selected_names)
    test_files = [entry["path"] for entry in candidates if _is_test_path(entry["path"])]
    references: list[dict] = []
    for path in test_files:
        try:
            source = store.resolve_evidence_file(repo_id, path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            references.append(_test_reference(path, "<file>", [], "UNKNOWN", "Test file could not be read."))
            continue
        try:
            module = ast.parse(source)
        except SyntaxError:
            names = re.findall(r"(?m)^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(", source)
            references.extend(_test_reference(path, name, [], "UNKNOWN", "Test syntax could not be analyzed.") for name in names)
            if not names:
                references.append(_test_reference(path, "<file>", [], "UNKNOWN", "Test syntax could not be analyzed."))
            continue
        imported_names = _imported_python_names(module)
        test_functions = [node for node in ast.walk(module)
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
        helpers = [node for node in ast.walk(module)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test_")]
        helper_calls = {node.name: _resolved_call_names(node, imported_names) for node in helpers}
        helper_names = set()
        while True:
            added = {name for name, calls in helper_calls.items()
                     if calls & (related_names | helper_names)} - helper_names
            if not added:
                break
            helper_names.update(added)
        for test_function in test_functions:
            calls = _resolved_call_names(test_function, imported_names)
            if not selected_names:
                references.append(_test_reference(path, test_function.name, [test_function.lineno], "UNKNOWN",
                                                  "The selected function could not be identified for static relationship analysis."))
                continue
            direct_lines = [node.lineno for node in ast.walk(test_function) if isinstance(node, ast.Call)
                            and _resolved_call_name(node, imported_names) in selected_names]
            if direct_lines or calls & selected_names:
                references.append(_test_reference(path, test_function.name, sorted(set(direct_lines or [test_function.lineno])),
                                                  "DIRECT", "Test calls the selected function directly."))
                continue
            indirect_lines = [node.lineno for node in ast.walk(test_function) if isinstance(node, ast.Call)
                              and _resolved_call_name(node, imported_names) in (related_names - selected_names)]
            if indirect_lines or calls & (related_names - selected_names) or calls & helper_names:
                references.append(_test_reference(path, test_function.name, sorted(set(indirect_lines or [test_function.lineno])),
                                                  "INDIRECT", "Test invokes a repository function connected to the selected code through static caller/dependency paths; this does not establish branch coverage."))
            else:
                references.append(_test_reference(path, test_function.name, [test_function.lineno], "UNRELATED",
                                                  "No relationship to the selected function was established by static call analysis."))
        if not test_functions:
            references.append(_test_reference(path, "<file>", [], "UNKNOWN", "No test functions could be identified in this discovered test file."))
    return {"test_files": test_files, "total_test_files_found": len(test_files), "references": references,
            "counts": {category: sum(item["classification"] == category for item in references)
                       for category in ("DIRECT", "INDIRECT", "UNRELATED", "UNKNOWN")},
            "coverage_status": "not_measured", "coverage_percent": None, "coverage_report_path": None,
            "coverage_unknown": True}


def _repository_related_symbols(repo_id: str, file_path: str, selected_source: str, start: int, end: int,
                                candidates: list[dict], store: RepositoryStore,
                                selected_names: set[str]) -> set[str]:
    """Trace only callers that execute a selected function, not its unused dependencies."""
    parsed_functions: list[tuple[str, set[str]]] = []
    for entry in candidates:
        path = entry["path"]
        if Path(path).suffix.lower() != ".py" or _is_test_path(path):
            continue
        try:
            source = selected_source if path == file_path else store.resolve_evidence_file(
                repo_id, path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            module = ast.parse(source)
        except SyntaxError:
            continue
        imported = _imported_python_names(module)
        parsed_functions.extend((node.name, _resolved_call_names(node, imported))
                                for node in ast.walk(module)
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    related = set(selected_names)
    while True:
        expanded = set(related)
        for name, calls in parsed_functions:
            if name not in related and calls & related:
                expanded.add(name)
        if expanded == related:
            return related
        related = expanded


def _coverage_measurement(repo_id: str, candidates: list[dict], store: RepositoryStore) -> dict:
    names = {"coverage.xml", "coverage.json"}
    paths = {entry["path"] for entry in candidates}
    repo = store.get(repo_id)
    tracked = run_git(repo, ["ls-files", "-z"], check=False).split("\0")
    untracked = run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"], check=False).split("\0")
    paths.update(path for path in tracked + untracked if path)
    for path in sorted(paths):
        if Path(path).name.lower() not in names:
            continue
        try:
            content = store.resolve_evidence_file(repo_id, path).read_text(encoding="utf-8", errors="replace")
            if path.lower().endswith(".xml"):
                root = ElementTree.fromstring(content)
                value = root.attrib.get("line-rate")
                percent = float(value) * 100 if value is not None else None
            else:
                data = json.loads(content)
                totals = data.get("totals", {}) if isinstance(data, dict) else {}
                value = totals.get("percent_covered", totals.get("percent"))
                percent = float(value) if value is not None else None
            if percent is not None and 0 <= percent <= 100:
                return {"coverage_status": "measured", "coverage_percent": round(percent, 2),
                        "coverage_report_path": path, "coverage_unknown": False}
        except (OSError, ValueError, TypeError, ElementTree.ParseError):
            continue
    return {"coverage_status": "not_measured", "coverage_percent": None,
            "coverage_report_path": None, "coverage_unknown": True}


def _test_reference(path: str, name: str, lines: list[int], classification: str, reason: str) -> dict:
    suffix = f"#{lines[0]}" if lines else "#file"
    return {"test_id": f"{path}::{name}{suffix}", "path": path, "test_names": [] if name == "<file>" else [name],
            "line_ranges": [str(line) for line in lines], "classification": classification,
            "relationship_reason": reason, "evidence_references": []}


def _rank_evidence(items: list[dict], test_discovery: dict) -> None:
    """Order evidence by investigation value and remap all generated references atomically."""
    priorities = {"source": 0, "source_function": 0, "related_function": 1, "test": 2,
                  "assignment": 3, "read": 3, "documentation": 4,
                  "commit": 5, "line_history": 5, "diff": 5, "blame": 5,
                  "dependency_configuration": 6, "test_configuration": 6}
    original_ids = {item["id"]: item for item in items}
    for item in items:
        item["relevance_rank"] = priorities.get(item["type"], 8)
    items.sort(key=lambda item: (item["relevance_rank"], item.get("path") or "", item.get("line_range") or "", item["id"]))
    remap = {}
    for index, item in enumerate(items, 1):
        old_id = item["id"]
        item["id"] = f"E{index:02d}"
        remap[old_id] = item["id"]
    for reference in test_discovery["references"]:
        reference["evidence_references"] = [remap[item] for item in reference["evidence_references"] if item in remap]


def _is_test_path(path: str) -> bool:
    candidate = Path(path)
    name = candidate.name.lower()
    return ("test" in name or "spec" in name or
            any(part.lower() in {"test", "tests", "spec", "specs"} for part in candidate.parts)) and candidate.suffix.lower() in {
                ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cs", ".go", ".rs"
            }


def _selected_function_relationships(source: str, start: int, end: int) -> tuple[set[str], set[str]]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return set(), set()
    functions = [node for node in ast.walk(module)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.lineno <= end and getattr(node, "end_lineno", node.lineno) >= start]
    selected_names = {node.name for node in functions}
    dependencies = set().union(*(_resolved_call_names(node, _imported_python_names(module)) for node in functions)) if functions else set()
    dependencies -= selected_names
    return selected_names, dependencies


def _imported_python_names(module: ast.Module) -> dict[str, str]:
    imported: dict[str, str] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = ".".join(filter(None, (node.module, alias.name)))
    return imported


def _resolved_call_name(node: ast.Call, imported: dict[str, str]) -> str | None:
    if isinstance(node.func, ast.Name):
        return imported.get(node.func.id, node.func.id).rsplit(".", 1)[-1]
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _resolved_call_names(node: ast.AST, imported: dict[str, str]) -> set[str]:
    return {resolved for call in ast.walk(node) if isinstance(call, ast.Call)
            if (resolved := _resolved_call_name(call, imported)) is not None}


def _relationship_records(path: str, entries: list[tuple[str, list[int]]]) -> list[dict]:
    records = []
    for test_name, lines in entries:
        line_ranges = [str(line) for line in lines]
        records.append({"path": path, "test_names": [test_name], "line_ranges": line_ranges,
                        "evidence_references": []})
    return records


def _test_excerpt(source: str, line_ranges: list[str]) -> str:
    lines = source.splitlines()
    numbers: set[int] = set()
    for value in line_ranges:
        try:
            line = int(value)
        except ValueError:
            continue
        numbers.update(range(max(1, line - 3), min(len(lines), line + 3) + 1))
    return "\n".join(f"{number}: {lines[number - 1]}" for number in sorted(numbers))[:1600]


def _skip(scope: dict, path: str, reason: str) -> None:
    scope["files_skipped_count"] += 1
    if len(scope["files_skipped"]) < 100:
        scope["files_skipped"].append({"path": path, "reason": reason})


def _rank_candidates(files: list[dict], selected_path: str, symbols: list[str], question: str,
                     dependency_depth: int, selected_text: str, read_file) -> list[dict]:
    """Prioritize the selected module, its imports, relevant references, tests, and docs."""
    selected = Path(selected_path)
    terms = {term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)}
    names = {symbol.lower() for symbol in symbols} | terms
    depth_by_path = {selected_path: 0}
    by_path = {entry["path"]: entry for entry in files}
    module_paths: dict[str, str] = {}
    for path in by_path:
        relative = Path(path).as_posix()
        if not relative.endswith(".py"):
            continue
        module_name = relative[:-3].removesuffix("/__init__").replace("/", ".")
        module_paths[module_name] = path
    queue = [(selected_path, selected_text, 0)]
    while queue:
        current_path, current_text, current_depth = queue.pop(0)
        if current_depth >= dependency_depth or Path(current_path).suffix.lower() != ".py":
            continue
        try:
            module = ast.parse(current_text)
        except SyntaxError:
            continue
        current_module = Path(current_path).as_posix()[:-3].removesuffix("/__init__").replace("/", ".")
        package = current_module if current_path.endswith("/__init__.py") else current_module.rpartition(".")[0]
        imported = []
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".") if package else []
                    trim = max(0, node.level - 1)
                    base = ".".join(parts[:len(parts) - trim] + ([base] if base else []))
                if base:
                    imported.append(base)
                if not node.module:
                    imported.extend(".".join(filter(None, (base, alias.name))) for alias in node.names)
        for dotted in imported:
            candidate = module_paths.get(dotted)
            if candidate is None:
                candidate = next((path for name, path in module_paths.items() if name.endswith("." + dotted)), None)
            if candidate and candidate not in depth_by_path:
                depth_by_path[candidate] = current_depth + 1
                try:
                    text = read_file(candidate)
                except OSError:
                    text = ""
                queue.append((candidate, text, current_depth + 1))
    def score(entry: dict) -> tuple[int, int, str]:
        path = entry["path"]
        lower = path.lower()
        if path == selected_path:
            return (0, 0, path)
        depth = depth_by_path.get(path)
        if depth is not None:
            return (1, depth, path)
        if any(token in lower for token in names):
            return (2, 0, path)
        is_test = "test" in Path(path).name.lower() or "spec" in Path(path).name.lower()
        if is_test and (selected.stem.lower() in lower or Path(path).parent == selected.parent):
            return (3, 0, path)
        if Path(path).name.lower().startswith(("readme", "contributing", "changelog")):
            return (4, 0, path)
        return (5, 0, path)
    return sorted(files, key=score)


def _path_for_commit(commit: dict, current_path: str) -> str:
    paths = [path for entry in commit.get("changed_paths", []) for path in entry.get("paths", [])]
    return current_path if current_path in paths else (paths[-1] if paths else current_path)


def _selected_symbols(file_path: str, lines: list[str], start: int, end: int, question: str = "") -> list[str]:
    symbols: list[str] = []
    if Path(file_path).suffix.lower() == ".py":
        try:
            module = ast.parse("\n".join(lines))
            for node in ast.walk(module):
                first = getattr(node, "lineno", 0)
                last = getattr(node, "end_lineno", first)
                if first <= end and last >= start:
                    if isinstance(node, ast.Name):
                        symbols.append(node.id)
                    elif isinstance(node, ast.Attribute):
                        symbols.append(node.attr)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        symbols.append(node.name)
                    elif isinstance(node, ast.arg):
                        symbols.append(node.arg)
            selected_functions = [node.name for node in ast.walk(module)
                                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                  and node.lineno <= end and getattr(node, "end_lineno", node.lineno) >= start]
            selected_calls = []
            for node in ast.walk(module):
                first = getattr(node, "lineno", 0)
                last = getattr(node, "end_lineno", first)
                if first <= end and last >= start and isinstance(node, ast.Call):
                    called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if called:
                        selected_calls.append(called)
            symbols = selected_functions + selected_calls + symbols
        except SyntaxError:
            pass
    if not symbols:
        selected = "\n".join(lines[start - 1:end])
        symbols = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", selected)
    seen: set[str] = set()
    filtered = []
    selected_file = "\n".join(lines)
    question_identifiers = [token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)
                            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", selected_file)]
    for symbol in question_identifiers + symbols:
        if symbol in keyword.kwlist or symbol.lower() in {"self", "true", "false", "none", "null", "return", "string", "value", "result"}:
            continue
        if symbol not in seen:
            seen.add(symbol)
            filtered.append(symbol)
        if len(filtered) >= MAX_SYMBOLS:
            break
    return filtered


def _add_symbol_context(file_path: str, text: str, symbols: list[str], add, matches: dict, *,
                        selected_path: str, selected_start: int, selected_end: int) -> None:
    if not symbols:
        return
    source_lines = text.splitlines()
    if Path(file_path).suffix.lower() == ".py":
        try:
            module = ast.parse(text)
        except SyntaxError:
            return
        parents = {child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)}
        if file_path == selected_path:
            selected_functions = [node for node in ast.walk(module)
                                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                  and node.lineno <= selected_end
                                  and getattr(node, "end_lineno", node.lineno) >= selected_start]
            selected_names = {node.name for node in selected_functions}
            imported = _imported_python_names(module)
            definitions = {node.name for node in ast.walk(module)
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            project_imports = {name for node in ast.walk(module)
                               if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.")
                               for alias in node.names for name in {alias.name, alias.asname or alias.name}}
            for caller in ast.walk(module):
                if not isinstance(caller, (ast.FunctionDef, ast.AsyncFunctionDef)) or caller.name in selected_names:
                    continue
                calls = _resolved_call_names(caller, imported)
                if not calls & selected_names:
                    continue
                line = caller.lineno
                key = (file_path, caller.name, "related_function")
                if not matches.get(key):
                    matches[key] = 1
                    first, last = line, getattr(caller, "end_lineno", line)
                    excerpt = "\n".join(f"{number}: {source_lines[number - 1]}" for number in range(first, last + 1))
                    targets = ", ".join(sorted(calls & selected_names))
                    add("related_function", f"{file_path}:{line} · caller {caller.name}", excerpt,
                        f"Python AST caller `{caller.name}` invokes selected function(s) {targets}; full caller body included.",
                        path=file_path, line_range=f"{first}-{last}")
                for call_name in _resolved_call_names(caller, imported):
                    if call_name in definitions or call_name in project_imports:
                        if call_name not in symbols:
                            symbols.append(call_name)
        for node in ast.walk(module):
            symbol = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                      else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None)
            if symbol not in symbols or not hasattr(node, "lineno"):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "related_function"
            else:
                kind = "assignment" if isinstance(node.ctx, (ast.Store, ast.Del)) else "read" if isinstance(node.ctx, ast.Load) else "related_function"
            key = (file_path, symbol, kind)
            if matches.get(key, 0) >= 1:
                continue
            matches[key] = matches.get(key, 0) + 1
            function = _enclosing_function(node, parents)
            line = node.lineno
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
                first, last = min([line, *decorator_lines]), getattr(node, "end_lineno", line)
                kind = "source_function" if file_path == selected_path and first <= selected_end and last >= selected_start else "related_function"
            else:
                first, last = max(1, line - 2), min(len(source_lines), line + 2)
            excerpt = "\n".join(f"{number}: {source_lines[number - 1]}" for number in range(first, last + 1))
            if function:
                excerpt = f"Within {function}:\n{excerpt}"
            add(kind, f"{file_path}:{line} · {symbol}", excerpt,
                f"Python AST {kind} of `{symbol}` in the inspected repository file.",
                path=file_path, line_range=f"{first}-{last}")
    else:
        for number, line in enumerate(source_lines, start=1):
            for symbol in symbols:
                if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", line):
                    continue
                assignment = bool(re.search(rf"\b(?:const|let|var)?\s*{re.escape(symbol)}\s*(?:=|:=|\+=|-=)", line))
                kind = "assignment" if assignment else "read"
                key = (file_path, symbol, kind)
                if matches.get(key, 0) >= 1:
                    continue
                matches[key] = matches.get(key, 0) + 1
                first, last = max(1, number - 2), min(len(source_lines), number + 2)
                excerpt = "\n".join(f"{n}: {source_lines[n - 1]}" for n in range(first, last + 1))
                add(kind, f"{file_path}:{number} · {symbol}", excerpt,
                    f"Text search match for `{symbol}`; assignment/read classification is lexical.",
                    path=file_path, line_range=f"{first}-{last}")


def _enclosing_function(node: ast.AST, parents: dict) -> str | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return current.name
        current = parents.get(current)
    return None
