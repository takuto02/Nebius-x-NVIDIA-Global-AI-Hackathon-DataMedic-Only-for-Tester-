"""
DataMedic agent: profiler -> Nemotron -> FixProposals -> safe repairs -> human decisions
---------------------------------------------------------------------------------------
Connects the profiler to NVIDIA Nemotron on Nebius Token Factory.

    1. The profiler finds the issues (free, runs locally)
    2. Nemotron receives a compact summary (issues + business rules + context)
       and replies with FixProposals (the team's agreed format)
    3. Code checks every proposal (guardrails) and applies only the SAFE ones.
       Each repair is re-checked; if it didn't work, it is rolled back.
    4. Risky proposals wait in pending_approvals.json for a human.

Every change is recorded in audit_log.json by the code that made it.

Setup (once):
    pip install openai
    Create a .env file next to this script:
        NEBIUS_API_KEY=your-key-here
        NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B

Usually you run this through run_datamedic.py, which creates a results folder.
All commands below take --dir <results folder> to work inside that folder.

Main run:
    python agent_demo.py data.csv --rules rules.json --dir results/run1
    python agent_demo.py data.csv --rules rules.json --dry-run      # show prompt, no credits

Human decisions (after a run):
    python agent_demo.py --pending --dir results/run1                          # what's waiting
    python agent_demo.py --approve --issue iss_001 --dir results/run1          # apply a proposed fix
    python agent_demo.py --resolve --issue iss_002 --row 63 --value 2026-02-03 --dir results/run1
    python agent_demo.py --accept  --issue iss_023 --dir results/run1          # keep data as it is
    python agent_demo.py --reject  --issue iss_023 --dir results/run1 --reason "wrong mapping"  # decline the fix

Other:
    python agent_demo.py --list-models
"""

import json
import os
import re
import sys

import profiler
from profiler import load_data, profile_dataset
from repair_tools import apply_repair, validate_repair, finalize_repair

# Row lists in issues are normally capped at 50 examples. Human decisions need
# the full list, so a row beyond the 50th can still be resolved and validated.
profiler.MAX_ROWS_PER_ISSUE = 10**9

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Used when the user gives no description of the dataset
CONTEXT = "No description was given. Work out what the data is from the column names and values."

# Repair tools the agent may choose from. Names/params must match repair_tools.py.
TOOLS = {
    "standardize_category": "params: {column, mapping: {raw_value: standard_value}}. "
                            "Copy raw values EXACTLY from the issue's `values` (keep spaces and case).",
    "parse_currency": "params: {column}. Converts '$1,200', '1200$', '1,200' to plain numbers.",
    "standardize_date": "params: {column}. Converts unambiguous dates to YYYY-MM-DD. "
                        "Use for mixed_date_formats and non_standard_date_format issues; "
                        "never use for ambiguous_date issues.",
    "trim_whitespace": "params: {column}. Removes leading/trailing spaces.",
    "remove_duplicate_rows": "params: {}. Deletes rows that exactly repeat an earlier row.",
    "flag_for_review": "params: {column, reason}. Makes no change; asks a human to decide.",
}

# Guardrails enforced by CODE, whatever the model says
ALWAYS_NEEDS_APPROVAL = {"ambiguous_date", "missing_values", "duplicate_key", "duplicate_rows",
                         "suspicious_negative", "outlier", "rule_violation", "future_date"}
DESTRUCTIVE_TOOLS = {"remove_duplicate_rows"}

# Which issue types each tool can actually fix (anything else becomes flag_for_review)
TOOL_FITS = {
    "standardize_category": {"category_variants"},
    "parse_currency": {"format_inconsistency"},
    "standardize_date": {"mixed_date_formats", "non_standard_date_format"},
    "trim_whitespace": {"whitespace"},
    "remove_duplicate_rows": {"duplicate_rows"},
}
AUTO_FIX_MIN_CONFIDENCE = 0.90

# File names inside the working folder
CLEANED, PENDING, AUDIT, PROPOSALS, RULES_USED = (
    "cleaned_output.csv", "pending_approvals.json", "audit_log.json", "proposals.json", "rules_used.json")

WORKDIR = "."


def wd(name):
    return os.path.join(WORKDIR, name)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_env(path=".env"):
    """Read KEY=value lines from .env into environment variables."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def client():
    from openai import OpenAI
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY is missing. Put it in a .env file next to this script.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def read_json(name, default):
    path = wd(name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(name, data):
    with open(wd(name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_rules():
    """--rules file if given, otherwise the rules saved with this run (if any)."""
    if arg("--rules"):
        with open(arg("--rules"), encoding="utf-8") as f:
            return json.load(f)
    return read_json(RULES_USED, None)


def same_value(a, b):
    """True if two cells mean the same thing ('450' vs 450.0 counts as the same)."""
    if str(a) == str(b):
        return True
    try:
        return float(str(a)) == float(str(b))
    except ValueError:
        return False


def cell_changes(before, after, column=None, limit=500):
    """What actually changed, cell by cell, so the audit log shows old -> new values."""
    if len(before) != len(after) or list(before.columns) != list(after.columns):
        return {"rows_removed": len(before) - len(after), "cells_changed": 0, "changes": []}
    cols = [column] if column in before.columns else list(before.columns)
    changes = []
    for col in cols:
        for i, (old, new) in enumerate(zip(before[col], after[col])):
            if not same_value(old, new):
                changes.append({"row": i, "column": col, "old_value": old, "new_value": new})
    return {"cells_changed": len(changes), "changes": changes[:limit]}


# ---------------------------------------------------------------------------
# Build the prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are DataMedic, a careful junior data engineer.
You receive data-quality issues found by a profiler, plus the user's business rules.
For each issue, propose ONE repair using only the allowed tools.

Allowed tools:
{tools}

Rules:
- Never invent data. Missing values, negative revenue, outliers, duplicate IDs, future dates
  and business-rule violations cannot be safely guessed: use flag_for_review.
- Ambiguous dates (like 03/04/26) must NOT be guessed: use flag_for_review.
- "key" must be copied from the issue's `key` field (e.g. "Province:category_variants:Ontario"),
  and "issue_id" from its `id` field (e.g. "iss_009"). Do not put the id in "key".
- For standardize_category, copy raw values exactly from the issue's `values` field.
- confidence is 0.0-1.0: 0.90+ high, 0.70-0.89 medium, below 0.70 low.
- needs_approval is true unless the fix is a safe, high-confidence formatting change.
- Several issues may describe the same rows; you may propose one fix per issue anyway.

Reply with ONLY a JSON object, no other text:
{{"proposals": [{{"key": "...", "issue_id": "...", "rule_id": null, "tool": "...",
  "params": {{...}}, "confidence": 0.0, "rationale": "...", "needs_approval": true}}]}}"""


def compact_issues(issues):
    """Send the model only what it needs (keeps prompts small and credits low)."""
    keep = ("id", "key", "column", "type", "severity", "row_count", "evidence", "values",
            "rule_id", "suggested_standard", "main_format", "expected_range")
    return [{k: i[k] for k in keep if k in i} for i in issues]


def build_messages(profile, rules, context):
    tools = "\n".join(f"- {name}: {desc}" for name, desc in TOOLS.items())
    user = {
        "dataset_context": context,
        "rows": profile["summary"]["rows"],
        "columns": [{"name": c["column"], "kind": c["kind"]} for c in profile["columns"]],
        "business_rules": rules or [],
        "issues": compact_issues(profile["issues"]),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools)},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=1)},
    ]


# ---------------------------------------------------------------------------
# Read and check the model's answer
# ---------------------------------------------------------------------------

def parse_json(text):
    """Pull the JSON object out of the reply, even if wrapped in ``` fences or extra text."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # drop reasoning, if any
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model reply contained no JSON object")
    return json.loads(text[start:end + 1])


def check_proposals(proposals, issues):
    """Guardrails. Returns (checked_proposals, warnings). Never trusts the model blindly."""
    by_key = {i["key"]: i for i in issues}
    by_id = {i["id"]: i for i in issues}
    checked, warnings = [], []

    for p in proposals:
        # Match on the stable key; fall back to the short id if the model mixed them up
        issue = (by_key.get(p.get("key")) or by_id.get(p.get("key"))
                 or by_id.get(p.get("issue_id")) or by_key.get(p.get("issue_id")))
        if issue is None:
            warnings.append(f"Dropped proposal for unknown issue {p.get('key')!r}")
            continue
        p["key"] = key = issue["key"]  # always store the real stable key
        if p.get("tool") not in TOOLS:
            warnings.append(f"{key}: unknown tool {p.get('tool')!r}, changed to flag_for_review")
            p["tool"], p["params"] = "flag_for_review", {"column": issue["column"],
                                                         "reason": "Model chose an invalid tool"}
        # Each tool only fixes certain kinds of issue (e.g. remove_duplicate_rows deletes
        # exact copies; it cannot fix a repeated ID or phone number)
        allowed = TOOL_FITS.get(p["tool"])
        if allowed is not None and issue["type"] not in allowed:
            warnings.append(f"{key}: {p['tool']} can't fix a {issue['type']} issue, changed to flag_for_review")
            p["tool"], p["params"] = "flag_for_review", {"column": issue["column"],
                                                         "reason": f"{p['tool']} does not fit this issue"}
            p["confidence"] = min(float(p.get("confidence", 0) or 0), 0.5)
        p["issue_id"] = issue["id"]
        p["rule_id"] = issue.get("rule_id")
        conf = float(p.get("confidence", 0) or 0)
        p["confidence"] = max(0.0, min(1.0, conf))

        # Mapping keys must be real values from the data (catches retyped/mangled values)
        incomplete = False
        if p["tool"] == "standardize_category":
            mapping = p.get("params", {}).get("mapping", {})
            real = issue.get("values", {})
            bad = [k for k in mapping if k not in real]
            if bad:
                warnings.append(f"{key}: mapping keys not found in data {bad}, removed")
                for k in bad:
                    mapping.pop(k)
            missing = [v for v in real if v not in mapping and v != issue.get("suggested_standard")]
            if missing:
                warnings.append(f"{key}: mapping skips values {missing}")
            incomplete = bool(bad or missing)

        # Approval policy enforced in code
        must_ask = (incomplete
                    or issue["type"] in ALWAYS_NEEDS_APPROVAL
                    or p["tool"] in DESTRUCTIVE_TOOLS
                    or p["confidence"] < AUTO_FIX_MIN_CONFIDENCE)
        if must_ask and not p.get("needs_approval"):
            warnings.append(f"{key}: model said auto-fix, policy requires approval")
        p["needs_approval"] = bool(p.get("needs_approval")) or must_ask
        checked.append(p)

    proposed = {p["key"] for p in checked}
    for i in issues:
        if i["key"] not in proposed:
            warnings.append(f"No proposal for issue {i['key']}")
    return checked, warnings


# ---------------------------------------------------------------------------
# Human decisions
# ---------------------------------------------------------------------------

def find_pending(pending, issue_id):
    return next((p for p in pending if p.get("issue_id") == issue_id), None)


def prune_pending(df, rules):
    """Remove pending items whose issue no longer exists (e.g. fixing negative revenue
    also clears the 'Revenue cannot be negative' rule). Each is logged, so the audit
    trail shows why it disappeared."""
    pending = read_json(PENDING, [])
    live = {i["key"] for i in profile_dataset(df, rules)["issues"]}
    cleared = [p for p in pending if p["key"] not in live]
    if not cleared:
        return
    audit_log = read_json(AUDIT, [])
    for p in cleared:
        audit_log.append({"tool": "resolved_by_other_fix", "column": p.get("params", {}).get("column"),
                          "issue_id": p.get("issue_id"), "key": p.get("key"), "status": "resolved",
                          "validated": True, "changed": False,
                          "rationale": "No longer detected after another fix or decision."})
        print(f"Also resolved: {p['key']} (no longer detected)")
    write_json(AUDIT, audit_log)
    write_json(PENDING, [p for p in pending if p["key"] in live])


def cmd_pending():
    pending = read_json(PENDING, [])
    if not pending:
        print("Nothing is waiting for a human decision.")
        return
    rules = load_rules()
    df = load_data(wd(CLEANED))
    issues = {i["key"]: i for i in profile_dataset(df, rules)["issues"]}
    print(f"{len(pending)} items waiting for a human decision:\n")
    for p in pending:
        live = issues.get(p["key"])
        where = f"{live['row_count']} rows, e.g. rows {live['rows'][:5]}" if live else "no longer detected"
        how = ("--approve (apply the proposed fix)" if p["tool"] != "flag_for_review"
               else "--resolve (enter a value per row) or --accept (keep as is)") + " or --reject (decline)"
        print(f"  {p['issue_id']}  {p['key']}")
        print(f"         proposed: {p['tool']}   affected: {where}")
        if live and live.get("values"):
            print(f"         values: {dict(list(live['values'].items())[:5])}")
        print(f"         decide with: {how}\n")


def cmd_accept():
    issue_id = arg("--issue")
    pending = read_json(PENDING, [])
    selected = find_pending(pending, issue_id)
    if selected is None:
        print(f"Pending issue not found: {issue_id}")
        return

    rules = load_rules()
    live = next((i for i in profile_dataset(load_data(wd(CLEANED)), rules)["issues"]
                 if i["key"] == selected["key"]), None)
    audit_log = read_json(AUDIT, [])
    audit_log.append({
        "tool": "human_acceptance",
        "column": selected.get("params", {}).get("column"),
        "issue_id": selected.get("issue_id"),
        "key": selected.get("key"),
        "status": "accepted_as_is",
        "rows_affected": live["row_count"] if live else 0,
        "validated": True,
        "changed": False,
        "rationale": "Reviewed by a human and accepted without modifying the data.",
    })
    write_json(AUDIT, audit_log)
    write_json(PENDING, [p for p in pending if p.get("issue_id") != issue_id])
    print(f"Accepted without changes: {selected['key']}")


def cmd_reject():
    """A human says NO to the proposed fix. The data is not changed, the issue stays
    in the data (and in the Health Score), and the decision is recorded in the audit log."""
    issue_id = arg("--issue")
    pending = read_json(PENDING, [])
    selected = find_pending(pending, issue_id)
    if selected is None:
        print(f"Pending issue not found: {issue_id}")
        return

    rules = load_rules()
    live = next((i for i in profile_dataset(load_data(wd(CLEANED)), rules)["issues"]
                 if i["key"] == selected["key"]), None)
    audit_log = read_json(AUDIT, [])
    audit_log.append({
        "tool": "human_rejection",
        "rejected_tool": selected.get("tool"),
        "column": selected.get("params", {}).get("column"),
        "issue_id": selected.get("issue_id"),
        "key": selected.get("key"),
        "status": "rejected_by_human",
        "rows_affected": live["row_count"] if live else 0,
        "validated": False,
        "changed": False,
        "rationale": arg("--reason") or "A human declined the proposed fix. The data was not changed.",
    })
    write_json(AUDIT, audit_log)
    write_json(PENDING, [p for p in pending if p.get("issue_id") != issue_id])
    print(f"Declined: {selected['key']} (proposed {selected.get('tool')}). Data not changed; "
          f"the issue stays open in the data.")


def cmd_resolve():
    issue_id, human_value, human_row = arg("--issue"), arg("--value"), arg("--row")
    if human_value is None or human_row is None:
        print("A human resolution needs --issue, --row and --value.")
        return
    human_row = int(human_row)

    pending = read_json(PENDING, [])
    selected = find_pending(pending, issue_id)
    if selected is None:
        print(f"Pending issue not found: {issue_id}")
        return

    rules = load_rules()
    df = load_data(wd(CLEANED))
    current = next((i for i in profile_dataset(df, rules)["issues"] if i["key"] == selected["key"]), None)
    if current is None:
        print(f"This issue is already resolved or no longer detected: {selected['key']}")
        return
    if human_row not in current.get("rows", []):
        print(f"Row {human_row} is not affected by {selected['key']}. "
              f"Affected rows include: {current['rows'][:10]}")
        return
    column = selected.get("params", {}).get("column") or current.get("column")
    if not column or column not in df.columns:
        print(f"Column not found: {column}")
        return

    def flags_on_cell(frame):
        """Issue keys that point at this row in this column (or at the whole row)."""
        return {i["key"] for i in profile_dataset(frame, rules)["issues"]
                if human_row in i.get("rows", []) and i.get("column") in (column, "*")}

    before_flags = flags_on_cell(df)
    old_value = df.iloc[human_row][column]
    df.at[df.index[human_row], column] = human_value

    # Validate THIS cell: the issue must no longer flag it, AND the new value must not
    # create a new problem (e.g. typing 'Feb 3rd' into a date column).
    after_issues = profile_dataset(df, rules)["issues"]
    after = next((i for i in after_issues if i["key"] == selected["key"]), None)
    after_flags = {i["key"] for i in after_issues
                   if human_row in i.get("rows", []) and i.get("column") in (column, "*")}
    new_problems = sorted(after_flags - before_flags)
    row_fixed = (after is None or human_row not in after.get("rows", [])) and not new_problems
    issue_gone = after is None

    audit_log = read_json(AUDIT, [])
    entry = {
        "tool": "human_resolution",
        "column": column,
        "row": human_row,
        "old_value": old_value,
        "new_value": human_value,
        "issue_id": selected.get("issue_id"),
        "key": selected.get("key"),
        "status": "applied" if row_fixed else "rejected",
        "validated": row_fixed,
    }
    if new_problems:
        entry["new_problems"] = new_problems
    audit_log.append(entry)
    write_json(AUDIT, audit_log)

    if not row_fixed:
        why = (f"it would create a new problem ({', '.join(new_problems)})" if new_problems
               else f"it still fails the check for {selected['key']}")
        print(f"Row {human_row}: '{human_value}' was NOT saved because {why}. "
              f"Try a value in the standard format (dates as YYYY-MM-DD, plain numbers).")
        return

    df.to_csv(wd(CLEANED), index=False)
    print(f"Row {human_row}: '{old_value}' -> '{human_value}'  (validated)")
    if issue_gone:
        write_json(PENDING, [p for p in pending if p.get("key") != selected["key"]])
        print(f"All rows fixed: {selected['key']} is resolved and removed from pending.")
    else:
        print(f"{after['row_count']} rows still need a decision for {selected['key']}.")
    prune_pending(df, rules)


def cmd_approve():
    issue_id = arg("--issue")
    pending = read_json(PENDING, [])
    selected = find_pending(pending, issue_id)
    if selected is None:
        print(f"Pending issue not found: {issue_id}")
        return
    if selected["tool"] == "flag_for_review":
        print(f"{issue_id} has no fix to approve. Use --resolve (enter values) or --accept (keep as is).")
        return

    rules = load_rules()
    current_df = load_data(wd(CLEANED))
    repaired, audit, before = apply_repair(current_df, selected, approved=True)
    resolved, _, audit = validate_repair(repaired, selected, audit, rules)
    current_df, audit = finalize_repair(repaired, before, audit)
    if audit["status"] == "applied":
        audit.update(cell_changes(before, current_df, audit.get("column")))

    current_df.to_csv(wd(CLEANED), index=False)
    audit_log = read_json(AUDIT, [])
    audit_log.append(audit)
    write_json(AUDIT, audit_log)
    print(f"Approved repair {issue_id} ({selected['key']}): {audit['status']} | validated={audit['validated']}")

    if audit["validated"]:
        write_json(PENDING, [p for p in pending if p.get("issue_id") != issue_id])
        prune_pending(current_df, rules)
        if audit.get("rows_removed"):
            print("Rows were removed, so row numbers have changed. Run --pending again before resolving.")
    else:
        print("The fix didn't solve the issue, so it was rolled back. It stays in pending.")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def main():
    global WORKDIR
    load_env()
    WORKDIR = arg("--dir") or "."
    os.makedirs(WORKDIR, exist_ok=True)

    for flag, command in (("--pending", cmd_pending), ("--accept", cmd_accept), ("--reject", cmd_reject),
                          ("--resolve", cmd_resolve), ("--approve", cmd_approve)):
        if flag in sys.argv:
            command()
            return

    if "--list-models" in sys.argv:
        for m in client().models.list().data:
            print(m.id)
        return

    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        return

    rules = load_rules()
    profile = profile_dataset(load_data(sys.argv[1]), rules)
    context = arg("--context") or CONTEXT

    if "--dry-run" in sys.argv:
        print(json.dumps(build_messages(profile, rules, context), indent=2, ensure_ascii=False))
        return

    model = os.environ.get("NEBIUS_MODEL")
    if not model:
        sys.exit("NEBIUS_MODEL is missing. Run with --list-models, then add it to .env")

    batch_size = 8
    all_proposals = []
    tokens_in = tokens_out = 0
    for start in range(0, len(profile["issues"]), batch_size):
        batch_profile = {**profile, "issues": profile["issues"][start:start + batch_size]}
        print(f"Asking {model} about {len(batch_profile['issues'])} issues...")
        resp = client().chat.completions.create(
            model=model,
            messages=build_messages(batch_profile, rules, context),
            temperature=0.1,
            max_tokens=8192,
        )
        reply = resp.choices[0].message.content
        if resp.usage:
            tokens_in += resp.usage.prompt_tokens
            tokens_out += resp.usage.completion_tokens
        try:
            all_proposals.extend(parse_json(reply)["proposals"])
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Could not read the model's reply ({e}). "
                  f"Finish reason: {resp.choices[0].finish_reason}. Raw reply:\n{reply}")
            return

    checked, warnings = check_proposals(all_proposals, profile["issues"])

    print(f"\n{len(checked)} proposals:")
    for p in checked:
        action = "NEEDS APPROVAL" if p["needs_approval"] else "auto-fix"
        print(f"  [{action:<14}] {p['key']:<40} {p['tool']:<22} conf {p['confidence']:.2f}")
        print(f"                   {p.get('rationale', '')}")
    if warnings:
        print("\nGuardrail warnings:")
        for w in warnings:
            print(f"  - {w}")
    if tokens_in or tokens_out:
        print(f"\nTokens used: {tokens_in} in + {tokens_out} out")

    write_json(PROPOSALS, {"proposals": checked, "warnings": warnings})

    # Apply only the safe proposals; everything else waits for a human
    current_df = load_data(sys.argv[1])
    audit_log, pending = [], []
    for proposal in checked:
        if proposal["needs_approval"]:
            pending.append(proposal)
            continue
        repaired, audit, before = apply_repair(current_df, proposal)
        resolved, _, audit = validate_repair(repaired, proposal, audit, rules)
        current_df, audit = finalize_repair(repaired, before, audit)
        if audit["status"] == "applied":
            audit.update(cell_changes(before, current_df, audit.get("column")))
        print(f"Repair {proposal['key']}: {audit['status']} | validated={audit['validated']}")
        audit_log.append(audit)

    current_df.to_csv(wd(CLEANED), index=False)
    write_json(AUDIT, audit_log)
    write_json(PENDING, pending)
    print(f"\nSaved {CLEANED}, {AUDIT}, {PENDING} and {PROPOSALS} in {os.path.abspath(WORKDIR)}")
    print(f"{len(pending)} items are waiting for a human decision.")


if __name__ == "__main__":
    main()
