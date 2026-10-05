"""
DataMedic synthetic dataset generator
-------------------------------------
Makes a realistic messy Canadian retail dataset, PLUS the answers:

    synthetic_messy.csv        the dirty data the agent has to fix
    synthetic_clean.csv        ground truth: what a perfect cleanup looks like
    synthetic_answer_key.csv   every planted error: row, column, type, bad value, correct value
    synthetic_rules.json       business rules for this dataset (DataMedic rule format)

Standards used in the clean file (match the Health Score):
    dates YYYY-MM-DD, full province names, plain numbers, no duplicates,
    genuinely missing values left blank (never invented).

Usage:
    python generate_dataset.py                    # 300 rows, seed 42
    python generate_dataset.py --rows 1000 --seed 7
    python generate_dataset.py --mess 3           # 3x more errors (good for a dramatic demo)

Same seed = same dataset every time, so the whole team can test on identical data.
"""

import csv
import json
import random
import sys
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

FIRST = ["Alice", "Ben", "Chloe", "David", "Emma", "Farid", "Grace", "Hugo", "Ivy", "Jack",
         "Kara", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rosa", "Sam", "Tara",
         "Omar", "Priya", "Lucas", "Sofia", "Ethan", "Aisha", "Mateo", "Hannah", "Wei", "Zoe"]
LAST = ["Martin", "Tremblay", "Singh", "Lee", "Roy", "Khan", "Chen", "Bouchard", "Wong", "Brown",
        "Patel", "Scott", "Nguyen", "Fraser", "Grant", "Dubois", "Adams", "Silva", "Gagnon", "Wilson",
        "Côté", "Campbell", "Ali", "MacDonald", "Kim", "Pelletier", "Taylor", "Morin", "Hassan", "Clark"]

# Province -> (weight, messy variants)
PROVINCES = {
    "Ontario": (38, ["ON", "Ont.", "ontario", "ONTARIO", "Ont"]),
    "Quebec": (22, ["QC", "Que.", "quebec", "Québec", "PQ"]),
    "British Columbia": (14, ["BC", "B.C.", "british columbia", "B.C"]),
    "Alberta": (12, ["AB", "Alta.", "alberta"]),
    "Manitoba": (4, ["MB", "Man.", "manitoba"]),
    "Saskatchewan": (3, ["SK", "Sask.", "saskatchewan"]),
    "Nova Scotia": (3, ["NS", "N.S.", "nova scotia"]),
    "New Brunswick": (2, ["NB", "N.B."]),
    "Newfoundland and Labrador": (1, ["NL", "Nfld."]),
    "Prince Edward Island": (1, ["PEI", "P.E.I."]),
}

# Product -> unit price (CAD)
PRODUCTS = {"Laptop": 1200, "Monitor": 225, "Keyboard": 60, "Mouse": 25,
            "Headphones": 150, "Webcam": 80, "Tablet": 550, "Docking Station": 180}

MISSING_TOKENS = ["", "N/A", "null", "-", "NA"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLUMNS = ["Order_ID", "Order_Date", "Customer", "Province", "Product", "Quantity", "Revenue"]

# How many of each error to plant, per 300 rows (scaled for other sizes)
ERROR_PLAN = {
    "province_variant": 30,
    "whitespace": 8,
    "currency_text": 20,
    "missing_value": 12,
    "mixed_date_format": 15,
    "ambiguous_date": 6,
    "negative_revenue": 4,
    "outlier": 3,
    "duplicate_row": 5,
    "duplicate_id": 3,
    # "Blind spot" errors the profiler isn't specifically built for
    "province_typo": 2,
    "zero_quantity": 2,
    "future_date": 2,
}

RULES = [
    {"id": "R1", "column": "Order_ID", "type": "unique", "description": "Order_ID must be unique"},
    {"id": "R2", "column": "Revenue", "type": "min_value", "value": 0,
     "description": "Revenue cannot be negative"},
    {"id": "R3", "column": "Province", "type": "canadian_province",
     "description": "Province must be a Canadian province"},
    {"id": "R4", "column": "Quantity", "type": "min_value", "value": 1,
     "description": "Quantity must be at least 1"},
    {"id": "R5", "column": "Order_ID", "type": "pattern", "value": "^ORD-\\d{5}$",
     "description": "Order_ID must look like ORD-12345"},
]


# ---------------------------------------------------------------------------
# Step 1: build clean, realistic data
# ---------------------------------------------------------------------------

def make_clean(n, rng):
    start = date(2026, 1, 1)
    names = list(PROVINCES)
    weights = [PROVINCES[p][0] for p in names]
    days = sorted(rng.randint(0, 180) for _ in range(n))  # orders spread over ~6 months
    rows = []
    for i in range(n):
        product = rng.choice(list(PRODUCTS))
        qty = rng.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
        rows.append({
            "Order_ID": f"ORD-{10001 + i}",
            "Order_Date": (start + timedelta(days=days[i])).isoformat(),
            "Customer": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "Province": rng.choices(names, weights=weights)[0],
            "Product": product,
            "Quantity": str(qty),
            "Revenue": str(qty * PRODUCTS[product]),
        })
    return rows


# ---------------------------------------------------------------------------
# Step 2: plant errors, recording every one
# ---------------------------------------------------------------------------

def plant_errors(clean, rng, scale):
    """Returns (messy_rows, truth_rows, answer_key).
    truth_rows = the ground truth (what a perfect cleanup produces)."""
    messy = [dict(r, _uid=i) for i, r in enumerate(clean)]
    truth = [dict(r) for r in clean]
    key = []
    used = set()  # (row uid, column) already corrupted: one error per cell

    def pick(column, condition=lambda r: True):
        pool = [r for r in messy if (r["_uid"], column) not in used and condition(r)]
        r = rng.choice(pool)
        used.add((r["_uid"], column))
        return r

    def record(r, column, etype, bad, correct, needs_human):
        key.append({"_uid": r["_uid"], "order_id": truth[r["_uid"]]["Order_ID"], "column": column,
                    "error_type": etype, "bad_value": bad, "correct_value": correct,
                    "needs_human": "yes" if needs_human else "no"})

    count = {k: max(1, round(v * scale)) for k, v in ERROR_PLAN.items()}

    for _ in range(count["province_variant"]):
        r = pick("Province")
        good = r["Province"]
        r["Province"] = rng.choice(PROVINCES[good][1])
        record(r, "Province", "province_variant", r["Province"], good, False)

    for _ in range(count["whitespace"]):
        r = pick("Province")
        good = r["Province"]
        r["Province"] = rng.choice(["  ", " "]) + good + rng.choice([" ", "  "])
        record(r, "Province", "whitespace", r["Province"], good, False)

    for _ in range(count["province_typo"]):
        r = pick("Province", lambda r: r["Province"] == "Ontario")
        r["Province"] = rng.choice(["Ontaro", "Onatrio"])
        record(r, "Province", "province_typo", r["Province"], "Ontario", False)

    for _ in range(count["currency_text"]):
        r = pick("Revenue", lambda r: int(r["Revenue"]) >= 1000)
        v = int(r["Revenue"])
        r["Revenue"] = rng.choice([f"${v:,}", f"{v}$", f"{v:,}", f"CAD {v}", f"${v}.00"])
        record(r, "Revenue", "currency_text", r["Revenue"], str(v), False)

    for _ in range(count["negative_revenue"]):
        r = pick("Revenue")
        good = r["Revenue"]
        r["Revenue"] = f"-{good}"
        record(r, "Revenue", "negative_revenue", r["Revenue"], good, True)

    for _ in range(count["outlier"]):
        r = pick("Revenue")
        good = r["Revenue"]
        r["Revenue"] = str(int(good) * 100)  # extra zeros typed by mistake
        record(r, "Revenue", "outlier", r["Revenue"], good, True)

    for _ in range(count["zero_quantity"]):
        r = pick("Quantity")
        good = r["Quantity"]
        r["Quantity"] = "0"
        record(r, "Quantity", "zero_quantity", "0", good, True)

    missing_cols = ["Revenue"] * 6 + ["Customer"] * 4 + ["Province"] * 2
    for i in range(count["missing_value"]):
        col = missing_cols[i % len(missing_cols)]
        r = pick(col)
        bad = rng.choice(MISSING_TOKENS)
        r[col] = bad
        truth[r["_uid"]][col] = ""  # can't be recovered -> perfect cleanup leaves it blank
        record(r, col, "missing_value", bad, "", True)

    for _ in range(count["mixed_date_format"]):
        # Unambiguous non-standard formats: day > 12, so day/month order is obvious.
        # Numeric dates are month-first, the SAME order used for the ambiguous ones below,
        # so a human (or the profiler clue) reading the column gets a consistent answer.
        r = pick("Order_Date", lambda r: int(r["Order_Date"][8:]) > 12)
        good = r["Order_Date"]
        y, m, d = good.split("-")
        r["Order_Date"] = rng.choice([f"{MONTHS[int(m) - 1]} {int(d)} {y}",
                                      f"{int(m)}/{d}/{y}",  # month-first, same as the ambiguous ones
                                      f"{MONTHS[int(m) - 1]} {int(d)}, {y}"])
        record(r, "Order_Date", "mixed_date_format", r["Order_Date"], good, False)

    for _ in range(count["ambiguous_date"]):
        # Both parts <= 12 and different: 03/04/26 could be March 4 or April 3
        r = pick("Order_Date", lambda r: int(r["Order_Date"][8:]) <= 12
                 and int(r["Order_Date"][8:]) != int(r["Order_Date"][5:7]))
        good = r["Order_Date"]
        y, m, d = good.split("-")
        r["Order_Date"] = f"{m}/{d}/{y[2:]}"  # true meaning is month-first
        record(r, "Order_Date", "ambiguous_date", r["Order_Date"], good, True)

    for _ in range(count["future_date"]):
        r = pick("Order_Date")
        good = r["Order_Date"]
        r["Order_Date"] = "2027" + good[4:]  # wrong year typed
        record(r, "Order_Date", "future_date", r["Order_Date"], good, True)

    for _ in range(count["duplicate_id"]):
        # A different order accidentally given an existing order's ID
        r = pick("Order_ID")
        other = rng.choice([o for o in messy if o["_uid"] != r["_uid"]])
        good = r["Order_ID"]
        r["Order_ID"] = other["Order_ID"]
        record(r, "Order_ID", "duplicate_id", r["Order_ID"], good, True)

    # Exact duplicate rows go in last (right after their original), since they add rows
    originals = rng.sample([r for r in messy if not any(k["_uid"] == r["_uid"] for k in key)],
                           count["duplicate_row"])
    for orig in originals:
        copy = dict(orig, _uid=f"dup-{orig['_uid']}")
        messy.insert(messy.index(orig) + 1, copy)
        key.append({"_uid": copy["_uid"], "order_id": orig["Order_ID"], "column": "*",
                    "error_type": "duplicate_row", "bad_value": "(whole row repeated)",
                    "correct_value": "(delete row)", "needs_human": "yes"})

    # Final row numbers (0-based, same as the profiler)
    position = {r["_uid"]: i for i, r in enumerate(messy)}
    for k in key:
        k["row"] = position[k.pop("_uid")]
    key.sort(key=lambda k: (k["row"], k["column"]))

    for r in messy:
        del r["_uid"]
    return messy, truth, key


# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    n = arg("--rows", 300)
    seed = arg("--seed", 42)
    mess = arg("--mess", 1.0)
    rng = random.Random(seed)

    clean = make_clean(n, rng)
    messy, truth, key = plant_errors(clean, rng, scale=n / 300 * mess)

    write_csv("synthetic_messy.csv", messy, COLUMNS)
    write_csv("synthetic_clean.csv", truth, COLUMNS)
    write_csv("synthetic_answer_key.csv", key,
              ["row", "order_id", "column", "error_type", "bad_value", "correct_value", "needs_human"])
    with open("synthetic_rules.json", "w", encoding="utf-8") as f:
        json.dump(RULES, f, indent=2)

    print(f"Generated {len(messy)} messy rows ({n} real orders + duplicates), seed {seed}, mess x{mess}")
    print(f"Planted {len(key)} errors:")
    counts = {}
    for k in key:
        counts[k["error_type"]] = counts.get(k["error_type"], 0) + 1
    for t, c in counts.items():
        print(f"  {t:<20} {c}")
    print("\nWrote synthetic_messy.csv, synthetic_clean.csv, "
          "synthetic_answer_key.csv, synthetic_rules.json")
