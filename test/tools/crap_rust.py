"""Compute per-function CRAP scores for Rust code and report offenders.

Rust sibling of crap.py (which is radon/coverage.py-bound and Python-only).

Reads:
  test-results/lizard-cc.csv       (lizard -l rust --csv output, no header)
  test-results/rust-coverage.json  (cargo llvm-cov ... --json, llvm.coverage.json.export)

Writes a table to stdout, sorted by CRAP desc. Exits 2 if any function has
CRAP > FAIL_THRESHOLD.

Formula: CRAP(f) = CC(f)^2 * (1 - cov(f)/100)^3 + CC(f)

Join strategy: llvm-cov identifies functions by mangled symbol + line regions;
lizard identifies them by source name + line span. They are joined per file by
line-span overlap, so no demangler is needed. Closures (separate llvm records
nested inside their enclosing fn's span) aggregate into the enclosing function.

Calibration notes (documented, not tunable):
- Coverage is llvm region coverage (executed code regions / total code regions),
  the same metric llvm-cov's own report calls "Regions Cover".
- lizard CCN counts match/if/&&/|| branches; scores are not directly comparable
  to radon's Python numbers but the thresholds are kept identical on purpose.
- Functions inside `#[cfg(test)]` mod blocks are excluded via a brace scan.
  A `#[cfg(test)] mod tests;` pointing at a separate file is NOT excluded —
  keep in-source unit tests in inline mod blocks (this repo's convention).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ALERT_THRESHOLD = 30.0
FAIL_THRESHOLD = 8.0
TARGET_THRESHOLD = 4.0

LIZARD_CSV = Path("test-results/lizard-cc.csv")
COVERAGE_JSON = Path("test-results/rust-coverage.json")

CODE_REGION = 0  # llvm-cov region kind for executable code
CFG_TEST_LOOKAHEAD = 3  # lines between #[cfg(test)] and its `mod ... {`


def crap(cc: float, cov_pct: float) -> float:
    gap = 1.0 - (cov_pct / 100.0)
    return cc * cc * (gap ** 3) + cc


def normalize(path: str) -> str:
    p = path.replace("\\", "/")
    root = str(Path.cwd()).replace("\\", "/").rstrip("/") + "/"
    if p.startswith(root):
        p = p[len(root):]
    return p.lstrip("./")


def cfg_test_ranges(source: str) -> list[tuple[int, int]]:
    """Line ranges (1-based, inclusive) of `#[cfg(test)] mod x { ... }` blocks.

    Brace counting ignores string/comment context; on unbalanced braces the
    scan fails open (no exclusion) rather than guessing.
    """
    lines = source.splitlines()
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "#[cfg(test)]":
            i += 1
            continue
        depth = 0
        opened = False
        end = None
        for j in range(i + 1, len(lines)):
            if not opened and j > i + CFG_TEST_LOOKAHEAD:
                break  # attribute gates something other than an inline mod
            if not opened and "mod" not in " ".join(lines[i + 1:j + 1]):
                continue
            depth += lines[j].count("{") - lines[j].count("}")
            if lines[j].count("{"):
                opened = True
            if opened and depth <= 0:
                end = j + 1
                break
        if end is not None:
            ranges.append((i + 1, end))
            i = end
        else:
            i += 1
    return ranges


def load_lizard(path: Path) -> list[dict]:
    """Rows: {file, name, cc, start, end} from lizard CSV (headerless).

    Columns: nloc, ccn, token, param, length, location, file, name, signature,
    start_line, end_line.
    """
    rows = []
    with path.open(newline="") as fh:
        for rec in csv.reader(fh):
            if len(rec) < 11:
                continue
            rows.append({
                "file": normalize(rec[6]),
                "name": rec[7],
                "cc": int(rec[1]),
                "start": int(rec[9]),
                "end": int(rec[10]),
            })
    excluded: dict[str, list[tuple[int, int]]] = {}
    kept = []
    for row in rows:
        if "/tests/" in f"/{row['file']}":
            continue
        if row["file"] not in excluded:
            src = Path(row["file"])
            excluded[row["file"]] = (
                cfg_test_ranges(src.read_text(encoding="utf-8", errors="replace"))
                if src.exists() else []
            )
        if any(lo <= row["start"] <= hi for lo, hi in excluded[row["file"]]):
            continue
        kept.append(row)
    return kept


def load_coverage(path: Path) -> dict[str, list[dict]]:
    """Return {normalized_file: [{start, end, executed, total}]} per llvm function."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for export in data.get("data", []):
        for fn in export.get("functions", []):
            filenames = fn.get("filenames", [])
            per_file: dict[int, dict] = {}
            for region in fn.get("regions", []):
                line_start, _, line_end, _, exec_count, file_id, _, kind = region[:8]
                if kind != CODE_REGION or file_id >= len(filenames):
                    continue
                slot = per_file.setdefault(
                    file_id, {"start": line_start, "end": line_end,
                              "executed": 0, "total": 0})
                slot["start"] = min(slot["start"], line_start)
                slot["end"] = max(slot["end"], line_end)
                slot["total"] += 1
                slot["executed"] += 1 if exec_count > 0 else 0
            for file_id, slot in per_file.items():
                out.setdefault(normalize(filenames[file_id]), []).append(slot)
    return out


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def collect_rows(lizard_rows: list[dict], coverage_map: dict) -> list[dict]:
    rows = []
    for fn in lizard_rows:
        executed = total = 0
        for span in coverage_map.get(fn["file"], []):
            if overlap(fn["start"], fn["end"], span["start"], span["end"]):
                executed += span["executed"]
                total += span["total"]
        cov_pct = (executed / total * 100.0) if total else 0.0
        rows.append({
            "file": fn["file"],
            "name": fn["name"],
            "cc": fn["cc"],
            "cov": cov_pct,
            "crap": crap(fn["cc"], cov_pct),
        })
    rows.sort(key=lambda r: r["crap"], reverse=True)
    return rows


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("CRAP Report (Rust): no functions analyzed")
        return
    print(f"{'CRAP':>7}  {'CC':>4}  {'Cov%':>6}  {'Function':<40}  File")
    print("-" * 100)
    for r in rows:
        print(f"{r['crap']:>7.1f}  {r['cc']:>4}  {r['cov']:>6.1f}  "
              f"{r['name']:<40}  {r['file']}")


def main() -> int:
    if not LIZARD_CSV.exists():
        print(f"CRAP check skipped — {LIZARD_CSV} not found")
        return 0
    lizard_rows = load_lizard(LIZARD_CSV)
    coverage_map = load_coverage(COVERAGE_JSON)
    rows = collect_rows(lizard_rows, coverage_map)
    print_report(rows)

    above_target = sum(1 for r in rows if r["crap"] > TARGET_THRESHOLD)
    above_fail = sum(1 for r in rows if r["crap"] > FAIL_THRESHOLD)
    above_alert = sum(1 for r in rows if r["crap"] > ALERT_THRESHOLD)
    print()
    print(f"Summary: {len(rows)} functions analyzed")
    print(f"  above target (>{TARGET_THRESHOLD}): {above_target}")
    print(f"  above fail   (>{FAIL_THRESHOLD}): {above_fail}")
    print(f"  above alert  (>{ALERT_THRESHOLD}): {above_alert}")
    return 2 if above_fail else 0


if __name__ == "__main__":
    sys.exit(main())
