import re
from datetime import datetime, timezone

from .repository_service import RepositoryStore, run_git

_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_LINE_HEADER = re.compile(r"^([0-9a-f]{40,64})\s+(\d+)\s+(\d+)(?:\s+(\d+))?$")
_META_FORMAT = "%H%x1f%P%x1f%s%x1f%an%x1f%aI"
MAX_COMMITS = 20


def file_history(store: RepositoryStore, repo_id: str, file_path: str,
                 start: int | None = None, end: int | None = None) -> dict:
    repo = store.get(repo_id)
    store.resolve_file(repo_id, file_path)
    head = run_git(repo, ["rev-parse", "--verify", "HEAD"], check=False).strip()
    commit_ids = run_git(repo, ["log", "--follow", f"-n{MAX_COMMITS}", "--format=%H", "--", file_path]).splitlines() if head else []
    commits: list[dict] = []
    diffs: list[dict] = []
    all_changed_paths: list[dict] = []

    for commit_id in commit_ids:
        if not _COMMIT.fullmatch(commit_id):
            continue
        meta = run_git(repo, ["show", "-s", f"--format={_META_FORMAT}", commit_id], check=False).strip()
        fields = meta.split("\x1f", 4)
        if len(fields) != 5 or fields[0] != commit_id:
            continue
        changed_paths = _changed_paths(repo, commit_id)
        commit = {
            "hash": fields[0], "short_hash": fields[0][:10],
            "parent": fields[1].split()[0] if fields[1].split() else None,
            "subject": fields[2], "author": fields[3], "date": fields[4],
            "changed_paths": changed_paths,
            "test_paths": [entry["path"] for entry in changed_paths
                           if "test" in entry["path"].lower() or "spec" in entry["path"].lower()],
        }
        commits.append(commit)
        all_changed_paths.extend({**entry, "commit": commit_id} for entry in changed_paths)

        if len(diffs) < 8 and changed_paths:
            paths = list(dict.fromkeys(path for entry in changed_paths for path in entry["paths"]))[:5]
            patch = run_git(repo, ["show", "--format=", "--no-ext-diff", "--find-renames", "--unified=10",
                                   commit_id, "--", *paths], timeout=8, check=False)
            if patch:
                diffs.append({"commit": commit_id, "parent": commit["parent"],
                              "changed_paths": changed_paths, "excerpt": patch[-9000:]})

    blame = _blame_range(repo, file_path, start, end) if start is not None and end is not None else []
    line_trace = _line_range_history(repo, file_path, start, end) if start is not None and end is not None and head else []
    shallow = run_git(repo, ["rev-parse", "--is-shallow-repository"], check=False).strip() == "true"
    selected_line_commits = list(dict.fromkeys(item["commit"] for item in blame))
    line_trace_commits = [item["hash"] for item in line_trace]

    remote = _public_remote(run_git(repo, ["remote", "get-url", "origin"], check=False).strip())
    if remote:
        for commit in commits:
            commit["url"] = f"{remote}/commit/{commit['hash']}"
        for diff in diffs:
            diff["url"] = f"{remote}/commit/{diff['commit']}"

    return {
        "commits": commits,
        "blame": blame,
        "diffs": diffs,
        "changed_paths": all_changed_paths,
        "line_trace": line_trace,
        "line_trace_commits": line_trace_commits,
        "selected_line_commits": selected_line_commits,
        "shallow": shallow,
        "insufficient_history": len(commits) < 2,
    }


def _changed_paths(repo, commit_id: str) -> list[dict]:
    raw = run_git(repo, ["diff-tree", "--root", "--no-commit-id", "--name-status", "-r",
                         "--find-renames", "-z", commit_id], check=False)
    cells = raw.split("\0")
    result: list[dict] = []
    index = 0
    while index < len(cells):
        status = cells[index].strip()
        index += 1
        if not status:
            continue
        needed = 2 if status.startswith(("R", "C")) else 1
        paths = [path for path in cells[index:index + needed] if path]
        index += needed
        if paths:
            result.append({"status": status, "path": paths[-1], "paths": paths})
    return result


def _blame_range(repo, file_path: str, start: int, end: int) -> list[dict]:
    end = min(end, start + 199)
    raw = run_git(repo, ["blame", "--line-porcelain", "-M", "-C", "-C",
                         f"-L{start},{end}", "--", file_path], timeout=15, check=False)
    records: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        header = _LINE_HEADER.match(line)
        if header:
            if current:
                records.append(current)
            current = {"commit": header.group(1), "original_line": int(header.group(2)),
                       "line": int(header.group(3)), "author": "", "date": "", "summary": "", "path": file_path}
        elif current is not None:
            if line.startswith("author "):
                current["author"] = line[7:]
            elif line.startswith("author-time "):
                current["date"] = _iso_timestamp(line[12:])
            elif line.startswith("summary "):
                current["summary"] = line[8:]
            elif line.startswith("filename "):
                current["path"] = line[9:].strip('"')
    if current:
        records.append(current)
    return records[:200]


def _iso_timestamp(raw: str) -> str:
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return raw


def _line_range_history(repo, file_path: str, start: int, end: int) -> list[dict]:
    end = min(end, start + 199)
    raw = run_git(repo, ["log", "--no-color", "-n12", f"--format={_META_FORMAT}",
                         f"-L{start},{end}:{file_path}"], timeout=18, check=False)
    result: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        if "\x1f" in line and _COMMIT.fullmatch(line.split("\x1f", 1)[0]):
            if current:
                result.append(current)
            fields = line.split("\x1f", 4)
            if len(fields) == 5:
                current = {"hash": fields[0], "parent": fields[1].split()[0] if fields[1].split() else None,
                           "subject": fields[2], "author": fields[3], "date": fields[4], "patch": ""}
            else:
                current = None
        elif current is not None and (line.startswith("@@") or line.startswith(("+", "-", " "))):
            current["patch"] += line + "\n"
    if current:
        result.append(current)
    return result


def _public_remote(value: str) -> str | None:
    # Link only recognized public Git hosts. Strip credentials from configured remotes.
    match = re.fullmatch(r"(?:https://|http://)?(?:[^/@]+@)?(github\.com|gitlab\.com)[:/]([^\s]+)", value)
    if not match:
        return None
    host, path = match.groups()
    path = path.removesuffix(".git").strip("/")
    if not path or ".." in path.split("/"):
        return None
    return f"https://{host}/{path}"
