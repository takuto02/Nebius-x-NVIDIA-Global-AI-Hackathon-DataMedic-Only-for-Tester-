"""
Check how many planted errors the profiler catches.

Compares the profiler's issues against an answer key (e.g. from generate_dataset.py):
a planted error counts as CAUGHT if some issue flags that same row and column.

Usage:
    python check_coverage.py synthetic_messy.csv synthetic_answer_key.csv synthetic_rules.json
"""

import csv
import json
import sys

import profiler
from profiler import load_data, profile_dataset

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

profiler.MAX_ROWS_PER_ISSUE = 10**9  # list every row, so nothing is missed by the 50-row cap

df = load_data(sys.argv[1])
rules = json.load(open(sys.argv[3], encoding="utf-8")) if len(sys.argv) > 3 else None
issues = profile_dataset(df, rules)["issues"]
key = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))

flagged = {}  # (row, column) -> list of issue types
for iss in issues:
    for row in iss["rows"]:
        flagged.setdefault((row, iss["column"]), []).append(iss["type"])


def caught_by(entry):
    row = int(entry["row"])
    if entry["column"] == "*":  # whole-row duplicate: any row-level flag counts
        return flagged.get((row, "*"), [])
    return flagged.get((row, entry["column"]), [])


by_type = {}
missed = []
for entry in key:
    hits = caught_by(entry)
    t = by_type.setdefault(entry["error_type"], {"planted": 0, "caught": 0})
    t["planted"] += 1
    if hits:
        t["caught"] += 1
    else:
        missed.append(entry)

total = sum(t["planted"] for t in by_type.values())
caught = sum(t["caught"] for t in by_type.values())
print(f"\nProfiler caught {caught} of {total} planted errors ({caught / total:.0%})\n")
print(f"  {'Error type':<20} {'Caught':>8}")
for name, t in by_type.items():
    mark = "" if t["caught"] == t["planted"] else "   <-- gap"
    print(f"  {name:<20} {t['caught']:>3} / {t['planted']:<3}{mark}")

if missed:
    print("\nMissed errors (blind spots to fix):")
    for m in missed:
        print(f"  row {m['row']:>4}  {m['column']:<11} {m['error_type']:<18} {m['bad_value']!r}")
