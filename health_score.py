"""
DataMedic Health Score (Role 4)
-------------------------------
Turns a dataset into a 0-100 Data Health Score across 5 dimensions:

    Dimension        Weight   What it measures
    Completeness      25%     % of cells that are not missing
    Uniqueness        20%     % of rows that are not duplicates (full row or repeated ID)
    Validity          20%     % of number/date cells already in a clean, standard format
    Consistency       15%     % of category cells written in one standard way
    Business rules    20%     % of rows that pass every user-defined rule

    Overall = weighted average of the dimensions.
    If a dimension can't be measured (e.g. no rules given, or no date/number
    columns), it is skipped and the remaining weights are scaled up to 100%.

Usage:
    python health_score.py messy_retail.csv --rules rules.json
    python health_score.py messy_retail.csv --rules rules.json --after clean_retail.csv
    python health_score.py messy_retail.csv --rules rules.json --json score.json
"""

import json
import re
import sys

import pandas as pd

from profiler import (
    VARIANT_TO_PROVINCE,
    canonical_value as canonical,
    is_missing,
    is_plain_number,
    load_data,
    looks_like_id,
    profile_column,
    rule_failures,
    split_rules,
    infer_rules,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

WEIGHTS = {
    "completeness": 0.25,
    "uniqueness": 0.20,
    "validity": 0.20,
    "consistency": 0.15,
    "business_rules": 0.20,
}

# The one date format we treat as "clean". Repair tools should convert to this.
STANDARD_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# Business rules are checked by rule_failures() in profiler.py (rule format documented there).


# ---------------------------------------------------------------------------
# Dimension scores. Each returns (passed, total, explanation) or None if N/A.
# ---------------------------------------------------------------------------

# Every dimension is measured the same way: the share of ROWS that pass.
# A record with even one bad value can break a report, so we judge rows, not cells.
# Each function returns (bad_rows_mask, explanation), or None if not measurable.

def no_rows(df):
    return pd.Series(False, index=df.index)


def score_completeness(df, kinds):
    missing = df.apply(lambda c: c.map(is_missing))
    bad = missing.any(axis=1)
    return bad, (f"{int(bad.sum())} of {len(df)} rows have a missing value "
                 f"({int(missing.values.sum())} missing cells)")


def score_uniqueness(df, kinds):
    normalized = df.apply(lambda c: c.map(lambda v: str(v).strip().lower()))
    bad = normalized.duplicated(keep="first")
    for col in df.columns:
        if looks_like_id(col):
            s = df[col]
            vals = s.map(lambda v: str(v).strip())
            bad |= ~s.map(is_missing) & vals.duplicated(keep="first")
    return bad, f"{int(bad.sum())} of {len(df)} rows are duplicates (full row or repeated ID)"


def score_validity(df, kinds):
    cols = [c for c, k in kinds.items() if k in ("numeric", "date")]
    if not cols:
        return None
    bad = no_rows(df)
    for col in cols:
        s = df[col]
        present = ~s.map(is_missing)
        if kinds[col] == "numeric":
            bad |= present & ~s.map(is_plain_number)
        else:
            bad |= present & ~s.map(lambda v: bool(STANDARD_DATE.match(str(v).strip())))
    return bad, (f"{int(bad.sum())} of {len(df)} rows have a number or date in a non-standard "
                 f"format (expected plain numbers and YYYY-MM-DD dates)")


def score_consistency(df, kinds):
    cols = [c for c, k in kinds.items() if k == "categorical"]
    if not cols:
        return None
    bad = no_rows(df)
    for col in cols:
        s = df[col][~df[col].map(is_missing)]
        for canon, values in s.groupby(s.map(canonical)):
            # Provinces: the full official name is the standard.
            # Anything else: the most common spelling is the standard.
            standard = canon if canon in VARIANT_TO_PROVINCE.values() else values.value_counts().index[0]
            wrong = (values != standard) | values.map(lambda v: str(v) != str(v).strip())
            bad.loc[values[wrong].index] = True
    return bad, (f"{int(bad.sum())} of {len(df)} rows have a category written in a "
                 f"non-standard way (spelling, abbreviation or extra spaces)")


def score_business_rules(df, kinds, rules):
    if not rules:
        return None
    bad = no_rows(df)
    per_rule = {}
    rules, _ = split_rules(df, rules)
    if not rules:
        return None
    for rule in rules:
        f = rule_failures(df, rule)
        per_rule[rule.get("id", rule["type"])] = int(f.sum())
        bad |= f
    broken = ", ".join(f"{rid}: {n}" for rid, n in per_rule.items() if n) or "none"
    return bad, f"{int(bad.sum())} of {len(df)} rows break a rule (rows per rule: {broken})"


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def grade(score):
    if score >= 95:
        return "Healthy"
    if score >= 80:
        return "Fair"
    if score >= 60:
        return "Poor"
    return "Critical"


def health_score(df, rules=None, kinds=None):
    """Score a dataset. `kinds` lets you reuse column types from a previous run,
    so before/after comparisons judge columns the same way."""
    if kinds is None:
        kinds = {c: profile_column(df, c)["kind"] for c in df.columns}

    raw = {
        "completeness": score_completeness(df, kinds),
        "uniqueness": score_uniqueness(df, kinds),
        "validity": score_validity(df, kinds),
        "consistency": score_consistency(df, kinds),
        "business_rules": score_business_rules(df, kinds, rules),
    }

    measured = {k: v for k, v in raw.items() if v is not None}
    weight_sum = sum(WEIGHTS[k] for k in measured)
    rows = len(df)

    dimensions = {}
    overall = 0.0
    for name, result in raw.items():
        if result is None:
            dimensions[name] = {"score": None, "weight": 0.0,
                                "explanation": "Not measured (nothing to check)"}
            continue
        bad_mask, explanation = result
        failed = int(bad_mask.sum())
        score = 100.0 * (rows - failed) / rows if rows else 100.0
        weight = WEIGHTS[name] / weight_sum
        overall += score * weight
        dimensions[name] = {"score": round(score, 1), "weight": round(weight, 3),
                            "failed_rows": failed, "total_rows": rows,
                            "failed_row_ids": [int(i) for i in bad_mask[bad_mask].index],
                            "explanation": explanation}

    overall = round(overall, 1)
    return {"overall": overall, "grade": grade(overall), "dimensions": dimensions, "kinds": kinds}


def compare(before, after):
    """Before/after comparison, plus any dimension that got worse (regression).

    A regression means MORE rows fail a check than before. We compare counts,
    not percentages: removing a duplicate row shrinks the total and can nudge a
    percentage down even though nothing got worse."""
    changes, regressions = {}, []
    for name in WEIGHTS:
        b, a = before["dimensions"][name], after["dimensions"][name]
        if b["score"] is None or a["score"] is None:
            continue
        changes[name] = {"before": b["score"], "after": a["score"],
                         "change": round(a["score"] - b["score"], 1),
                         "failed_rows_before": b["failed_rows"], "failed_rows_after": a["failed_rows"]}
        if a["failed_rows"] > b["failed_rows"]:
            regressions.append(name)
    return {
        "overall": {"before": before["overall"], "after": after["overall"],
                    "change": round(after["overall"] - before["overall"], 1)},
        "dimensions": changes,
        "regressions": regressions,  # non-empty -> tell the agent to roll back
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

LABELS = {"completeness": "Completeness", "uniqueness": "Uniqueness", "validity": "Validity",
          "consistency": "Consistency", "business_rules": "Business rules"}


def print_score(title, result):
    print(f"\n{title}: {result['overall']}/100  ({result['grade']})")
    for name, d in result["dimensions"].items():
        if d["score"] is None:
            print(f"  {LABELS[name]:<15}   --    {d['explanation']}")
        else:
            print(f"  {LABELS[name]:<15} {d['score']:>5.1f}   {d['explanation']}")


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    before_df = load_data(sys.argv[1])
    rules = None
    if arg("--rules") == "auto":
        rules = infer_rules(before_df)
    elif arg("--rules"):
        with open(arg("--rules"), encoding="utf-8") as f:
            rules = json.load(f)
        if rules and not split_rules(before_df, rules)[0]:
            print("Note: none of these rules fit this dataset, so automatic rules are used instead.")
            rules = infer_rules(before_df)
    before = health_score(before_df, rules)
    print_score("Health Score", before)
    output = {"before": before}

    if arg("--after"):
        after = health_score(load_data(arg("--after")), rules, kinds=before["kinds"])
        print_score("Health Score after repair", after)
        cmp = compare(before, after)
        print(f"\nChange: {cmp['overall']['before']} -> {cmp['overall']['after']} "
              f"({cmp['overall']['change']:+})")
        if cmp["regressions"]:
            print(f"WARNING: these dimensions got worse: {', '.join(cmp['regressions'])}")
        else:
            print("No regressions: no dimension got worse.")
        output.update({"after": after, "comparison": cmp})

    if arg("--json"):
        with open(arg("--json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {arg('--json')}")
