import pandas as pd
from profiler import profile_dataset, parse_date, is_ambiguous_date

def trim_whitespace(df, column):
    """Remove leading and trailing whitespace from values in one column."""
    result = df.copy()

    if column not in result.columns:
        raise ValueError(f"Column not found: {column}")

    result[column] = result[column].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    return result

def standardize_category(df, column, mapping):
    """Replace exact raw category values using the agent-provided mapping."""
    result = df.copy()

    if column not in result.columns:
        raise ValueError(f"Column not found: {column}")

    result[column] = result[column].replace(mapping)

    return result

def parse_currency(df, column):
    """Convert currency-formatted strings into plain numeric values."""
    result = df.copy()
    result[column] = result[column].astype(object)

    if column not in result.columns:
        raise ValueError(f"Column not found: {column}")

    cleaned = (
        result[column]
        .astype(str)
        .str.strip()
        .str.replace("CAD", "", regex=False)
        .str.replace("USD", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    numeric = pd.to_numeric(cleaned, errors="coerce")

    valid = numeric.notna()
    # Write whole numbers without a trailing ".0" ("2400", not "2400.0")
    result.loc[valid, column] = numeric[valid].map(
        lambda n: str(int(n)) if float(n).is_integer() else str(n))

    return result

def standardize_date(df, column):
    """Convert unambiguous dates to YYYY-MM-DD format.

    Uses the profiler's date reader, which understands both month-first
    (11/27/2021) and day-first (27/11/2021) dates, plus 'Jan 15 2026'.
    Ambiguous dates like 4/2/2021 are NEVER converted: they stay as they are
    for a human to decide."""
    result = df.copy()

    if column not in result.columns:
        raise ValueError(f"Column not found: {column}")

    def convert_date(value):
        if not isinstance(value, str) or not value.strip():
            return value
        if is_ambiguous_date(value):
            return value  # could be two different dates: leave for human review
        parsed = parse_date(value)
        return parsed.isoformat() if parsed else value

    result[column] = result[column].map(convert_date)

    return result


def remove_duplicate_rows(df):
    """Remove rows that exactly duplicate an earlier row."""
    result = df.copy()

    result = result.drop_duplicates(keep="first").reset_index(drop=True)

    return result

def flag_for_review(df, column, reason):
    """Leave the data unchanged and record that human review is required."""
    result = df.copy()

    if column not in result.columns:
        raise ValueError(f"Column not found: {column}")

    return result

REPAIR_TOOLS = {
    "trim_whitespace": trim_whitespace,
    "standardize_category": standardize_category,
    "parse_currency": parse_currency,
    "standardize_date": standardize_date,
    "remove_duplicate_rows": remove_duplicate_rows,
    "flag_for_review": flag_for_review,
}

def apply_repair(df, proposal, approved=False):
    if proposal.get("needs_approval", False) and not approved:
        raise PermissionError("This repair requires human approval.")
    
    tool_name = proposal.get("tool")
    params = proposal.get("params", {})
    if tool_name not in REPAIR_TOOLS:
        raise ValueError(f"Unknown repair tool: {tool_name}")
    
    before = df.copy(deep=True)
    
    tool_function = REPAIR_TOOLS[tool_name]
    result = tool_function(df, **params)
    after = result.copy(deep=True)
    changed = not before.equals(after)
    
    audit = {
        "tool": tool_name,
        "column": params.get("column"),
        "issue_id": proposal.get("issue_id"),
        "key": proposal.get("key"),
        "rule_id": proposal.get("rule_id"),
        "confidence": proposal.get("confidence"),
        "rationale": proposal.get("rationale"),
        "needs_approval": proposal.get("needs_approval", False),
        "approved": approved if proposal.get("needs_approval", False) else None,
        "status": "applied",
        "changed": changed,
        "validated": False,
    }

    return result, audit, before


def rollback_repair(before, audit):
    restored = before.copy(deep=True)
    audit["status"] = "rolled_back"
    return restored, audit


def validate_repair(repaired_df, proposal, audit, rules=None):
    new_profile = profile_dataset(repaired_df, rules)
    issue_key = proposal.get("key")

    remaining_keys = {
        issue.get("key")
        for issue in new_profile.get("issues", [])
    }

    resolved = issue_key not in remaining_keys
    audit["validated"] = resolved

    if not resolved:
        audit["status"] = "validation_failed"

    return resolved, new_profile, audit


def finalize_repair(repaired_df, before, audit):
    if audit["validated"]:
        return repaired_df, audit

    restored, audit = rollback_repair(before, audit)
    return restored, audit