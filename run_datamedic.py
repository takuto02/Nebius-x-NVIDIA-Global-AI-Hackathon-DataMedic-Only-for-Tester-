"""
Run the whole DataMedic pipeline with ONE command
-------------------------------------------------
    1. Profile the dataset                          (profiler.py)
    2. Nemotron proposes fixes, safe ones applied   (agent_demo.py)
    3. Audit report PDF                             (audit_to_pdf.py)
    4. Health Score before vs after                 (health_score.py)

Every run gets its OWN results folder, so nothing is overwritten and your
original file is never changed:

    results/Messy_Employee_dataset_2026-09-30_151200/
        cleaned_output.csv       the cleaned data
        audit_report.pdf         the audit report
        audit_log.json           every change, recorded by the code that made it
        pending_approvals.json   risky items waiting for a human
        proposals.json           Nemotron's proposed fixes
        profile.json             every issue the profiler found
        health_score.json        Health Score before vs after
        rules_used.json          the business rules used
        run_info.json            which dataset this run belongs to

Run from the Validation folder (where the scripts and .env are).

Usage:
    python run_datamedic.py ../Data/any_file.csv                          # automatic rules
    python run_datamedic.py ../Data/messy_retail.csv --rules rules.json   # your own rules
    python run_datamedic.py ../Data/titanic.csv --context "Titanic passenger list, 1912"

After making human decisions (see the commands printed at the end of a run),
rebuild the report and score for that run:
    python run_datamedic.py --report results/<run folder>
"""

import json
import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import time
from datetime import datetime

PY = sys.executable


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def step(number, title, command, required=True):
    print(f"\n{'=' * 64}\nSTEP {number}: {title}\n{'=' * 64}")
    start = time.time()
    ok = subprocess.run(command).returncode == 0
    print(f"--> {'done' if ok else 'FAILED'} ({time.time() - start:.0f}s)")
    if not ok and required:
        sys.exit(f"\nStopped: step {number} failed. Fix the error above, then run again.")
    return ok


def report(out_dir, dataset):
    """Steps 3 and 4: PDF report + Health Score, for a run folder."""
    rules_file = os.path.join(out_dir, "rules_used.json")
    cleaned = os.path.join(out_dir, "cleaned_output.csv")
    step(3, "Create the audit report PDF",
         [PY, "audit_to_pdf.py", "--dir", out_dir, "--dataset", os.path.basename(dataset)], required=False)
    if os.path.exists(cleaned):
        step(4, "Health Score: before vs after",
             [PY, "health_score.py", dataset, "--rules", rules_file, "--after", cleaned,
              "--json", os.path.join(out_dir, "health_score.json")], required=False)
    else:
        print("\nSTEP 4 skipped: no cleaned file was created, so there's nothing to compare.")


def summary(out_dir, dataset):
    print(f"\n{'=' * 64}\nDATAMEDIC FINISHED: {os.path.basename(dataset)}\n{'=' * 64}")
    print(f"Your original file was NOT changed. Everything from this run is in:\n  {out_dir}{os.sep}\n")
    files = {
        "cleaned_output.csv": "the cleaned data",
        "audit_report.pdf": "the audit report",
        "audit_log.json": "every change, as data",
        "pending_approvals.json": "risky items waiting for a human",
        "proposals.json": "Nemotron's proposed fixes",
        "profile.json": "every issue the profiler found",
        "health_score.json": "Health Score before vs after",
        "rules_used.json": "the business rules used",
    }
    for name, desc in files.items():
        if os.path.exists(os.path.join(out_dir, name)):
            print(f"  ✓ {name:<26} {desc}")

    pending_path = os.path.join(out_dir, "pending_approvals.json")
    pending = json.load(open(pending_path, encoding="utf-8")) if os.path.exists(pending_path) else []
    if pending:
        d = out_dir
        print(f"\n{len(pending)} items are waiting for a human decision. Next steps:")
        print(f"  See them:        python agent_demo.py --pending --dir {d}")
        print(f"  Apply a fix:     python agent_demo.py --approve --issue <id> --dir {d}")
        print(f"  Enter a value:   python agent_demo.py --resolve --issue <id> --row <n> --value <v> --dir {d}")
        print(f"  Keep as is:      python agent_demo.py --accept  --issue <id> --dir {d}")
        print(f"  Decline a fix:   python agent_demo.py --reject  --issue <id> --dir {d}")
        print(f"  Then update the report and score:  python run_datamedic.py --report {d}")


def main():
    # ---- Rebuild report + score after human decisions ----
    if "--report" in sys.argv:
        out_dir = arg("--report")
        info_path = os.path.join(out_dir or "", "run_info.json")
        if not out_dir or not os.path.exists(info_path):
            sys.exit("Give the results folder of a run, e.g. --report results/messy_retail_2026-09-30_151200")
        dataset = json.load(open(info_path, encoding="utf-8"))["dataset"]
        report(out_dir, dataset)
        summary(out_dir, dataset)
        return

    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        sys.exit(1)

    dataset = sys.argv[1]
    rules = arg("--rules")
    if not os.path.exists(dataset):
        sys.exit(f"Can't find the dataset '{dataset}'. Check the path (e.g. ../Data/messy_retail.csv).")
    if rules and not os.path.exists(rules):
        sys.exit(f"Can't find the rules file '{rules}'.")
    for script in ("profiler.py", "agent_demo.py", "repair_tools.py", "audit_to_pdf.py", "health_score.py"):
        if not os.path.exists(script):
            sys.exit(f"Can't find {script}. Run this from the Validation folder, where the DataMedic scripts are.")
    if not os.path.exists(".env"):
        sys.exit("Can't find .env (your Nebius API key). It must be in this folder.")

    name = os.path.splitext(os.path.basename(dataset))[0]
    out_dir = os.path.join("results", f"{name}_{datetime.now():%Y-%m-%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Choose the business rules (never crash on a mismatch) ----
    from profiler import infer_rules, load_data, split_rules

    try:
        data = load_data(dataset)
    except pd.errors.EmptyDataError:
        st.error(
            "The uploaded CSV is empty and contains no columns or data. "
            "Please upload a CSV containing at least one column."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

    chosen = None
    if rules:
        usable, skipped = split_rules(data, json.load(open(rules, encoding="utf-8")))
        for s in skipped:
            print(f"  (skipping rule {s})")
        if usable:
            chosen = usable
            print(f"Using {len(usable)} rules from {rules}")
        else:
            print(f"None of the rules in {rules} fit this dataset, so DataMedic will make its own.")
    if chosen is None:
        chosen = infer_rules(data)
        print(f"Using {len(chosen)} automatic business rules:")
        for r in chosen:
            print(f"  {r['id']:<4} {r['description']}")

    rules_file = os.path.join(out_dir, "rules_used.json")
    json.dump(chosen, open(rules_file, "w", encoding="utf-8"), indent=2)
    json.dump({"dataset": os.path.abspath(dataset), "context": arg("--context"), "started": datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8"), indent=2)

    step(1, "Profile the data",
         [PY, "profiler.py", dataset, "--rules", rules_file, "--json", os.path.join(out_dir, "profile.json")])
    agent_cmd = [PY, "agent_demo.py", dataset, "--rules", rules_file, "--dir", out_dir]
    if arg("--context"):
        agent_cmd += ["--context", arg("--context")]
    step(2, "Nemotron proposes fixes, safe fixes are applied", agent_cmd)
    report(out_dir, dataset)
    summary(out_dir, dataset)


if __name__ == "__main__":
    main()
