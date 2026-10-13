"""
DataMedic Profiler (Role 4)
---------------------------
Looks at a messy CSV/Excel dataset, describes every column, and turns anything
suspicious into Issues in the team's shared format:

    {"id", "key", "column", "rows", "row_count", "type", "severity",
     "evidence", "values", "rule_id"}

    id       - short label for this run (iss_001, iss_002, ...). Changes between runs.
    key      - stable label that stays the same across runs, e.g.
               "Province:category_variants:Ontario". Use this to track an issue
               from profiling -> fix -> re-test.
    rows     - example row numbers (capped at 50). row_count has the real total.
               Repair tools should fix by matching `values`, not only these rows.
    values   - the exact raw values involved and how often each appears, so the
               agent never has to copy values out of the evidence sentence.
    rule_id  - the business rule this issue breaks (e.g. "R2"), or null.

The profiler NEVER changes the data. It only describes it. Fixing is done by
the repair tools (Role 3) after the agent (Role 2) decides what to do.

Usage:
    python profiler.py messy_retail.csv                                  # summary
    python profiler.py messy_retail.csv --rules rules.json               # + business rules
    python profiler.py messy_retail.csv --rules rules.json --json out.json   # save full profile
"""

import json
import re
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Strings that really mean "missing"
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing"}

# How many example rows to include per issue (keeps output readable)
MAX_ROWS_PER_ISSUE = 50

# A column counts as "mostly numeric" / "mostly dates" above this share
TYPE_THRESHOLD = 0.6

# Reference data: Canadian provinces and territories, with common variants.
# Used to spot values that mean the same province written different ways.
PROVINCE_VARIANTS = {
    "Ontario": ["ontario", "on", "ont", "ont."],
    "Quebec": ["quebec", "québec", "qc", "que", "que.", "pq"],
    "British Columbia": ["british columbia", "bc", "b.c.", "b.c"],
    "Alberta": ["alberta", "ab", "alta", "alta."],
    "Manitoba": ["manitoba", "mb", "man", "man."],
    "Saskatchewan": ["saskatchewan", "sk", "sask", "sask."],
    "Nova Scotia": ["nova scotia", "ns", "n.s.", "n.s"],
    "New Brunswick": ["new brunswick", "nb", "n.b.", "n.b"],
    "Newfoundland and Labrador": ["newfoundland and labrador", "newfoundland", "nl", "nfld", "nfld."],
    "Prince Edward Island": ["prince edward island", "pei", "pe", "p.e.i."],
    "Northwest Territories": ["northwest territories", "nt", "nwt"],
    "Nunavut": ["nunavut", "nu"],
    "Yukon": ["yukon", "yt", "yk"],
}
VARIANT_TO_PROVINCE = {v: p for p, vs in PROVINCE_VARIANTS.items() for v in vs}

# Date patterns we recognize. Two-part-ambiguous ones (dd/mm vs mm/dd) are
# checked separately so we can flag them instead of guessing.
DATE_REGEXES = {
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"),
    "NN/NN/YYYY": re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    "NN/NN/YY": re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$"),
    "NN-NN-YYYY": re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),
    "Mon D, YYYY": re.compile(r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"),
    "D Mon YYYY": re.compile(r"^\d{1,2} [A-Za-z]{3,9}\.? \d{4}$"),
}

CURRENCY_CHARS = re.compile(r"[$€£¥,\s]|CAD|USD", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path):
    """Load CSV or Excel as all-text so we see values exactly as written."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        try:
            # utf-8-sig also removes the invisible marker Excel adds when saving a CSV
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def is_missing(value):
    """Return True only for genuinely missing values or recognized missing tokens."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()

    return bool(text in MISSING_TOKENS)


def to_number(value):
    """'$1,200' / '1200$' / ' 1200 ' / '(50)' -> float, or None if not a number."""
    if is_missing(value):
        return None
    s = str(value).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = CURRENCY_CHARS.sub("", s.strip("()"))
    try:
        n = float(s)
        return -n if negative else n
    except ValueError:
        return None


def is_plain_number(value):
    """True only if the value is already a clean number like 1200 or -3.5."""
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", str(value).strip()))


def date_format_of(value):
    s = str(value).strip()
    for name, rx in DATE_REGEXES.items():
        if rx.match(s):
            return name
    return None


def is_ambiguous_date(value):
    """03/04/26 could be March 4 or April 3. True when both parts are <= 12
    and different, so the day/month order can't be known from the value."""
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$", str(value).strip())
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(2))
    return a <= 12 and b <= 12 and a != b


def normalize_key(value):
    """Lowercase, trim, drop punctuation. 'Ont.' and ' ont ' -> 'ont'."""
    return re.sub(r"[^\w\s]", "", str(value).strip().lower())


def canonical_value(value):
    """Group key for 'same value, different spelling'. Provinces map to their
    official name ('ON' -> 'Ontario'); anything else to its normalized form."""
    key = normalize_key(value)
    return VARIANT_TO_PROVINCE.get(key) or VARIANT_TO_PROVINCE.get(str(value).strip().lower()) or key


def value_counts(series, limit=50):
    """Exact raw values -> count, as a plain dict (keeps spaces and case)."""
    return {str(k): int(v) for k, v in series.value_counts().head(limit).items()}


def shape_of(value):
    """Turn a value into a pattern: 'ORD-0012' -> 'AAA-9999'."""
    s = str(value).strip()
    s = re.sub(r"[A-Za-z]", "A", s)
    s = re.sub(r"\d", "9", s)
    return s


def looks_like_id(col_name):
    name = col_name.lower().replace(" ", "_")
    return name == "id" or name.endswith("_id") or name.endswith("id") or "number" in name


def row_list(index_values):
    """Convert pandas index to plain row numbers (0-based), capped."""
    return [int(i) for i in list(index_values)[:MAX_ROWS_PER_ISSUE]]


# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

def looks_like_code_column(non_missing, unique_ratio):
    """Mostly-numeric columns that are really codes, like tickets ('349909', 'PC 17599').
    The non-numeric values mix letters and digits, and values are mostly unique."""
    if unique_ratio <= 0.5:
        return False
    not_numbers = non_missing[non_missing.map(to_number).isnull()]
    if len(not_numbers) == 0:
        return False
    mixed = not_numbers.map(lambda v: bool(re.search(r"[A-Za-z]", v)) and bool(re.search(r"\d", v)))
    return mixed.mean() >= 0.8


def infer_kind(non_missing, numeric_share, date_share, unique_ratio, col_name):
    if len(non_missing) == 0:
        return "empty"
    if looks_like_id(col_name) and unique_ratio > 0.8:
        return "identifier"
    if date_share >= TYPE_THRESHOLD:
        return "date"
    if numeric_share >= TYPE_THRESHOLD and looks_like_code_column(non_missing, unique_ratio):
        return "code"
    if numeric_share >= TYPE_THRESHOLD:
        return "numeric"
    # Categorical = values repeat a lot. Names/free text are mostly unique.
    if unique_ratio < 0.5 or (non_missing.nunique() <= 30 and unique_ratio < 0.9):
        return "categorical"
    return "text"


def profile_column(df, col):
    s = df[col]
    missing_mask = s.map(is_missing)
    non_missing = s[~missing_mask]
    n = len(non_missing)

    numbers = non_missing.map(to_number)
    numeric_share = numbers.notnull().mean() if n else 0.0
    date_formats = non_missing.map(date_format_of)
    date_share = date_formats.notnull().mean() if n else 0.0
    unique_ratio = non_missing.nunique() / n if n else 0.0

    kind = infer_kind(non_missing, numeric_share, date_share, unique_ratio, col)

    profile = {
        "column": col,
        "kind": kind,
        "rows": int(len(s)),
        "missing_count": int(missing_mask.sum()),
        "missing_rate": round(float(missing_mask.mean()), 4),
        "unique_count": int(non_missing.nunique()),
        "unique_ratio": round(float(unique_ratio), 4),
        "numeric_share": round(float(numeric_share), 4),
        "date_share": round(float(date_share), 4),
        "top_values": {str(k): int(v) for k, v in non_missing.value_counts().head(10).items()},
        "top_patterns": {k: int(v) for k, v in non_missing.map(shape_of).value_counts().head(5).items()},
    }

    if kind == "numeric":
        nums = numbers.dropna()
        profile["stats"] = {
            "min": float(nums.min()), "max": float(nums.max()),
            "mean": round(float(nums.mean()), 2), "median": float(nums.median()),
        }
    if kind == "date":
        profile["date_formats"] = {k: int(v) for k, v in date_formats.value_counts().items()}

    return profile


# ---------------------------------------------------------------------------
# Issue detectors. Each returns a list of issues (without ids yet).
# ---------------------------------------------------------------------------

def issue(column, rows, type_, severity, evidence, values=None, key_suffix=None,
          rule_id=None, **extra):
    key = f"{column}:{type_}" + (f":{key_suffix}" if key_suffix else "")
    out = {"key": key, "column": column, "rows": row_list(rows), "row_count": len(rows),
           "type": type_, "severity": severity, "evidence": evidence,
           "values": values or {}, "rule_id": rule_id}
    out.update(extra)
    return out


def detect_missing(df, col, prof):
    rows = df.index[df[col].map(is_missing)]
    if len(rows) == 0:
        return []
    rate = prof["missing_rate"]
    severity = "high" if rate > 0.2 else "medium" if rate > 0.05 else "low"
    tokens = sorted({repr(str(v)) for v in df.loc[rows, col]})[:5]
    return [issue(col, rows, "missing_values", severity,
                  f"{len(rows)} missing values ({rate:.1%}); written as {', '.join(tokens)}",
                  values=value_counts(df.loc[rows, col]))]


def detect_numeric_format(df, col, prof):
    """Numeric column where some values are strings like '$1,200' or '1200$'."""
    if prof["kind"] != "numeric":
        return []
    s = df[col]
    mask = s.map(lambda v: not is_missing(v) and not is_plain_number(v) and to_number(v) is not None)
    bad_mask = s.map(lambda v: not is_missing(v) and to_number(v) is None)
    out = []
    if mask.any():
        examples = s[mask].unique()[:5].tolist()
        out.append(issue(col, df.index[mask], "format_inconsistency", "medium",
                         f"{int(mask.sum())} numeric values stored as formatted text, e.g. {examples}",
                         values=value_counts(s[mask])))
    if bad_mask.any():
        examples = s[bad_mask].unique()[:5].tolist()
        out.append(issue(col, df.index[bad_mask], "invalid_type", "high",
                         f"{int(bad_mask.sum())} values in a numeric column can't be read as numbers, e.g. {examples}",
                         values=value_counts(s[bad_mask])))
    return out


def detect_negatives_and_outliers(df, col, prof):
    if prof["kind"] != "numeric":
        return []
    nums = df[col].map(to_number)
    out = []

    neg = (nums < 0).fillna(False)
    if neg.any() and neg.mean() < 0.2:  # a few negatives in a mostly-positive column
        out.append(issue(col, df.index[neg], "suspicious_negative", "medium",
                         f"{int(neg.sum())} negative values in a mostly positive column "
                         f"(possible business-rule violation, e.g. refunds or data entry errors)",
                         values=value_counts(df.loc[neg, col])))

    clean = nums.dropna()
    if len(clean) >= 10:
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - 3 * iqr, q3 + 3 * iqr
            outl = ((nums < low) | (nums > high)).fillna(False)
            if outl.any():
                examples = df.loc[outl, col].unique()[:5].tolist()
                out.append(issue(col, df.index[outl], "outlier", "low",
                                 f"{int(outl.sum())} extreme values outside [{low:,.2f}, {high:,.2f}], e.g. {examples}",
                                 values=value_counts(df.loc[outl, col]),
                                 expected_range=[round(float(low), 2), round(float(high), 2)]))
    return out


def detect_category_variants(df, col, prof):
    """Same category written different ways: 'ontario' / 'ON' / 'Ont.'"""
    if prof["kind"] not in ("categorical", "text"):
        return []
    s = df[col]
    non_missing = s[~s.map(is_missing)]
    if non_missing.empty:
        return []

    groups = non_missing.groupby(non_missing.map(canonical_value))
    out = []
    for canon, values in groups:
        spellings = values.unique()
        if len(spellings) > 1:
            counts = values.value_counts()
            is_province = canon in PROVINCE_VARIANTS
            # Provinces: the official full name is the standard. Otherwise the most common spelling.
            standard = canon if is_province else str(counts.index[0])
            rows = values[values != standard].index
            evidence = (f"{len(spellings)} spellings of the same value: "
                        f"{counts.to_dict()}"
                        + (f" -> likely '{canon}' (Canadian province)" if is_province
                           else f" -> most common is '{standard}'"))
            out.append(issue(col, rows, "category_variants", "medium", evidence,
                             values=value_counts(values), key_suffix=standard,
                             suggested_standard=standard))
    return out


def detect_whitespace(df, col, prof):
    s = df[col]
    mask = s.map(lambda v: not is_missing(v) and str(v) != str(v).strip())
    if not mask.any():
        return []
    return [issue(col, df.index[mask], "whitespace", "low",
                  f"{int(mask.sum())} values have leading/trailing spaces",
                  values=value_counts(s[mask]))]


def detect_dates(df, col, prof):
    if prof["kind"] != "date":
        return []
    s = df[col]
    out = []

    formats = prof.get("date_formats", {})
    if len(formats) > 1:
        main = max(formats, key=formats.get)
        mask = s.map(lambda v: not is_missing(v) and date_format_of(v) not in (None, main))
        out.append(issue(col, df.index[mask], "mixed_date_formats", "medium",
                         f"Dates use {len(formats)} formats: {formats}; most common is {main}",
                         values=value_counts(s[mask]), main_format=main))

    amb = s.map(lambda v: not is_missing(v) and is_ambiguous_date(v))
    if amb.any():
        examples = s[amb].unique()[:5].tolist()
        out.append(issue(col, df.index[amb], "ambiguous_date", "high",
                         f"{int(amb.sum())} dates can't be read safely (day/month order unknown), "
                         f"e.g. {examples}. Needs human review, do not guess.",
                         values=value_counts(s[amb])))

    unparseable = s.map(lambda v: not is_missing(v) and date_format_of(v) is None)
    if unparseable.any():
        examples = s[unparseable].unique()[:5].tolist()
        out.append(issue(col, df.index[unparseable], "invalid_type", "high",
                         f"{int(unparseable.sum())} values in a date column aren't recognizable dates, e.g. {examples}",
                         values=value_counts(s[unparseable])))
    return out


def parse_date(value):
    """Read a date in any format we recognize. Returns a date, or None when the
    format is unknown OR ambiguous (we never guess day/month order)."""
    s = str(value).strip().replace(",", "").replace(".", "")
    if is_ambiguous_date(s):
        return None
    for fmt in ("%Y-%m-%d", "%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})$", s)
    if m:  # unambiguous only: whichever part is > 12 must be the day
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        day, month = (a, b) if a > 12 else (b, a)
        try:
            return datetime(y, month, day).date()
        except ValueError:
            return None
    return None


def detect_future_dates(df, col, prof):
    """Dates later than today: usually a typo in the year (2027 instead of 2026)."""
    if prof["kind"] != "date":
        return []
    today = datetime.now().date()
    s = df[col]
    mask = s.map(lambda v: not is_missing(v) and (parse_date(v) or today) > today)
    if not mask.any():
        return []
    examples = s[mask].unique()[:5].tolist()
    return [issue(col, df.index[mask], "future_date", "medium",
                  f"{int(mask.sum())} dates are later than today ({today.isoformat()}), "
                  f"e.g. {examples}. Possibly a mistyped year.",
                  values=value_counts(s[mask]))]


def detect_duplicate_keys(df, col, prof):
    """ID-like column that should be unique but has repeats."""
    if not looks_like_id(col):
        return []
    s = df[col]
    non_missing = s[~s.map(is_missing)].map(lambda v: str(v).strip())
    dup = non_missing[non_missing.duplicated(keep=False)]
    if dup.empty:
        return []
    examples = dup.value_counts().head(5).to_dict()
    return [issue(col, dup.index, "duplicate_key", "high",
                  f"{dup.nunique()} IDs appear more than once ({len(dup)} rows), e.g. {examples}",
                  values=value_counts(dup))]


COLUMN_DETECTORS = [
    detect_missing,
    detect_numeric_format,
    detect_negatives_and_outliers,
    detect_category_variants,
    detect_whitespace,
    detect_dates,
    detect_future_dates,
    detect_duplicate_keys,
]


def detect_duplicate_rows(df):
    normalized = df.apply(lambda col: col.map(lambda v: str(v).strip().lower()))
    mask = normalized.duplicated(keep="first")
    if not mask.any():
        return []
    return [issue("*", df.index[mask], "duplicate_rows", "high",
                  f"{int(mask.sum())} rows are exact copies of an earlier row (ignoring case/spaces)")]


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------
# Rules are checked by code, not by the AI: exact, repeatable, and free.
# The agent receives violations as ready-made issues and only decides how to fix them.
#
# Rule format (JSON list). Supported types:
#   {"id": "R1", "column": "Order_ID", "type": "unique"}
#   {"id": "R2", "column": "Revenue",  "type": "not_null"}
#   {"id": "R3", "column": "Revenue",  "type": "min_value", "value": 0}
#   {"id": "R4", "column": "Quantity", "type": "max_value", "value": 1000}
#   {"id": "R5", "column": "Product",  "type": "allowed_values", "values": ["Laptop", "Mouse"]}
#   {"id": "R6", "column": "Province", "type": "canadian_province"}
#   {"id": "R7", "column": "Order_ID", "type": "pattern", "value": "^ORD-\\d{4}$"}
# Every rule may also have a "description" in plain English for the audit trail.

def rule_failures(df, rule):
    """Return a boolean Series: True where the row breaks this rule."""
    col = rule["column"]
    if col not in df.columns:
        raise ValueError(f"Rule {rule.get('id')} refers to column '{col}', but the file only has "
                         f"these columns: {list(df.columns)}. Check you're using the right "
                         f"rules file for this dataset.")
    s = df[col]
    present = ~s.map(is_missing)
    kind = rule["type"]

    if kind == "unique":
        values = s.map(lambda v: str(v).strip())
        return present & values.duplicated(keep="first")
    if kind == "not_null":
        return ~present
    if kind == "min_value":
        nums = s.map(to_number)
        return (present & nums.notnull() & (nums < rule["value"])).fillna(False)
    if kind == "max_value":
        nums = s.map(to_number)
        return (present & nums.notnull() & (nums > rule["value"])).fillna(False)
    if kind == "allowed_values":
        allowed = {str(v).strip().lower() for v in rule["values"]}
        return present & ~s.map(lambda v: str(v).strip().lower() in allowed)
    if kind == "canadian_province":
        return present & ~s.map(lambda v: canonical_value(v) in PROVINCE_VARIANTS)
    if kind == "pattern":
        rx = re.compile(rule["value"])
        return present & ~s.map(lambda v: bool(rx.fullmatch(str(v).strip())))
    raise ValueError(f"Unknown rule type '{kind}' in rule {rule.get('id')}")


def detect_rule_violations(df, rules):
    out = []
    for rule in rules or []:
        mask = rule_failures(df, rule)
        if not mask.any():
            continue
        rid = rule.get("id", rule["type"])
        col = rule["column"]
        desc = rule.get("description", f"{col} {rule['type']}")
        out.append(issue(col, df.index[mask], "rule_violation", "high",
                         f"{int(mask.sum())} rows break rule {rid}: \"{desc}\"",
                         values=value_counts(df.loc[mask, col]), key_suffix=rid, rule_id=rid))
    return out


# ---------------------------------------------------------------------------
# Automatic business-rule inference
# ---------------------------------------------------------------------------

def infer_rules(df):
    """
    Infer conservative business rules from the dataset itself.

    These rules are only used when the user did not provide a rules JSON.
    Rules are based on column names and observed data patterns rather than
    assuming a particular dataset schema.
    """
    rules = []

    for col in df.columns:
        s = df[col]
        non_missing = s[~s.map(is_missing)]

        if non_missing.empty:
            continue

        col_lower = str(col).strip().lower().replace(" ", "_")

        # ID-like columns should normally be unique.
        if looks_like_id(col):
            rules.append({
                "id": f"AUTO_{len(rules) + 1}",
                "column": col,
                "type": "unique",
                "description": f"{col} should contain unique identifiers."
            })

        # Canadian province columns can be checked against the known
        # province/territory vocabulary.
        if "province" in col_lower:
            rules.append({
                "id": f"AUTO_{len(rules) + 1}",
                "column": col,
                "type": "canadian_province",
                "description": f"{col} should contain a valid Canadian province or territory."
            })

        # Financial/amount columns should not normally contain negative
        # values. This is deliberately limited to clearly monetary fields.
        if any(word in col_lower for word in (
            "revenue", "amount", "price", "cost", "sales", "income",
            "expense", "profit", "fare", "salary", "payment"
        )):
            numbers = non_missing.map(to_number).dropna()

            if len(numbers) > 0 and (numbers >= 0).mean() >= 0.8:
                rules.append({
                    "id": f"AUTO_{len(rules) + 1}",
                    "column": col,
                    "type": "min_value",
                    "value": 0,
                    "description": f"{col} should not contain negative values."
                })

    return rules


def split_rules(df, rules):
    """
    Separate rules that apply to this dataset from rules that do not.

    This allows a user to provide a rules JSON created for another dataset
    without causing DataMedic to crash.
    """
    usable = []
    skipped = []

    if not rules:
        return usable, skipped

    if isinstance(rules, dict):
        rules = rules.get("rules", [])

    supported_types = {
        "unique",
        "not_null",
        "min_value",
        "max_value",
        "allowed_values",
        "canadian_province",
        "pattern",
    }

    for rule in rules:
        if not isinstance(rule, dict):
            skipped.append(str(rule))
            continue

        rule_id = rule.get("id", "UNKNOWN")
        column = rule.get("column")
        rule_type = rule.get("type")

        if not column or column not in df.columns:
            skipped.append(rule_id)
            continue

        if rule_type not in supported_types:
            skipped.append(rule_id)
            continue

        # Validate required fields for rule types that need them.
        if rule_type in {"min_value", "max_value"} and "value" not in rule:
            skipped.append(rule_id)
            continue

        if rule_type == "allowed_values" and "values" not in rule:
            skipped.append(rule_id)
            continue

        if rule_type == "pattern" and "value" not in rule:
            skipped.append(rule_id)
            continue

        usable.append(rule)

    return usable, skipped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile_dataset(df, rules=None):
    """Returns {"summary", "columns", "issues"}, all JSON-serializable.
    Pass `rules` (a list of rule dicts) to include business-rule violations."""
    columns = [profile_column(df, c) for c in df.columns]
    issues = []
    for prof in columns:
        for detector in COLUMN_DETECTORS:
            issues.extend(detector(df, prof["column"], prof))
    issues.extend(detect_duplicate_rows(df))
    issues.extend(detect_rule_violations(df, rules))

    # Keys must be unique so they can be tracked across runs
    seen = {}
    for iss in issues:
        n = seen.get(iss["key"], 0)
        seen[iss["key"]] = n + 1
        if n:
            iss["key"] = f"{iss['key']}#{n + 1}"

    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (severity_order[i["severity"]], i["column"], i["key"]))
    for n, iss in enumerate(issues, start=1):
        issues[n - 1] = {"id": f"iss_{n:03d}", **iss}

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = sum(c["missing_count"] for c in columns)
    summary = {
        "profiled_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(missing_cells),
        "missing_cell_rate": round(missing_cells / total_cells, 4) if total_cells else 0.0,
        "issue_count": len(issues),
        "issues_by_severity": {s: sum(1 for i in issues if i["severity"] == s) for s in severity_order},
        "issues_by_type": pd.Series([i["type"] for i in issues]).value_counts().to_dict() if issues else {},
    }
    return {"summary": summary, "columns": columns, "issues": issues}


def print_report(result):
    s = result["summary"]
    print(f"\nDataset: {s['rows']} rows x {s['columns']} columns")
    print(f"Missing cells: {s['missing_cells']} ({s['missing_cell_rate']:.1%})")
    print(f"Issues found: {s['issue_count']}  {s['issues_by_severity']}\n")
    print("Columns:")
    for c in result["columns"]:
        print(f"  - {c['column']:<15} {c['kind']:<12} missing {c['missing_rate']:.0%}, "
              f"{c['unique_count']} unique")
    print("\nIssues:")
    for i in result["issues"]:
        print(f"  [{i['severity'].upper():<6}] {i['id']} {i['key']}")
        print(f"           {i['evidence']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rules = None
    if "--rules" in sys.argv:
        with open(sys.argv[sys.argv.index("--rules") + 1], encoding="utf-8") as f:
            rules = json.load(f)
    data = load_data(sys.argv[1])
    result = profile_dataset(data, rules)
    print_report(result)
    if "--json" in sys.argv:
        out_path = sys.argv[sys.argv.index("--json") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull profile saved to {out_path}")
