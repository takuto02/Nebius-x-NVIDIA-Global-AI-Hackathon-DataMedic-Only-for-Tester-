"""
DataMedic web app
-----------------
A web page on top of the DataMedic pipeline:

    upload a CSV  ->  profile + Nemotron fixes  ->  human decisions  ->  Health Score + audit PDF

It runs the same scripts as the command line (run_datamedic.py and agent_demo.py),
so every result is identical and saved in its own results/ folder.
Every number on the page is read from that folder: nothing is made up for show.

Setup (once), from the Validation folder:
    pip install streamlit

Start it:
    streamlit run app.py
"""

import json
import os
import subprocess
import sys
from html import escape

import pandas as pd
import streamlit as st

import profiler
from profiler import load_data, profile_dataset

profiler.MAX_ROWS_PER_ISSUE = 10**9  # human decisions need every affected row

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
UPLOADS = os.path.join(HERE, "uploads")

LABELS = {"completeness": "Completeness", "uniqueness": "Uniqueness", "validity": "Validity",
          "consistency": "Consistency", "business_rules": "Business rules"}
HUMAN_TOOLS = ("human_resolution", "human_acceptance", "human_rejection")
NOT_REPAIRS = HUMAN_TOOLS + ("resolved_by_other_fix",)

st.set_page_config(page_title="DataMedic", page_icon="✚", layout="wide")


# ---------------------------------------------------------------------------
# Look and feel (colors and fonts from the team's web design)
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#f6f8fb;--ink:#0b1220;--muted:#687486;--line:#e6eaf0;--navy:#0c1422;--blue:#4f6fff;--blue2:#eef1ff;
      --green:#16a874;--green2:#e8f8f2;--amber:#d68a12;--amber2:#fff5df;--red:#d95757;--red2:#fff0f0}
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button{font-family:'DM Sans',sans-serif}
[data-testid="stIconMaterial"],.material-symbols-rounded{font-family:'Material Symbols Rounded'!important}
.stApp{background:var(--bg)}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.02em;color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1400px}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1320 0%,#101a2a 100%)}
[data-testid="stSidebar"] *{color:#c9d3e1}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"]{background-color:#16233a!important;border-color:#243249!important}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input{color:#e6ecf5!important;-webkit-text-fill-color:#e6ecf5}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:rgba(255,255,255,.06);border:1px solid #243249}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover{border-color:#4f6fff}

/* Buttons */
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(100deg,#4f6fff,#6d7fff);border:0;color:#fff;font-weight:700;
  box-shadow:0 10px 24px rgba(79,111,255,.2)}
[data-testid="stBaseButton-primary"]:hover{box-shadow:0 14px 30px rgba(79,111,255,.28);color:#fff}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"]{
  border-radius:10px;border:1px solid var(--line);font-weight:600}
.stButton button,.stDownloadButton button,.stFormSubmitButton button{border-radius:10px}

/* Cards, tabs, inputs */
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line)!important;border-radius:17px;
  box-shadow:0 8px 28px rgba(20,31,51,.03)}
[data-baseweb="tab-highlight"]{background:var(--blue)}
[data-baseweb="tab"][aria-selected="true"] p{color:var(--blue)}
[data-baseweb="tab"] p{font-weight:600}
textarea,[data-baseweb="input"]{border-radius:11px!important}
[data-testid="stFileUploaderDropzone"]{background:#fbfcfe;border:1.5px dashed #d7dee8;border-radius:16px}
[data-testid="stExpander"] details{border-radius:12px;border-color:var(--line)}

/* Custom blocks */
.dm-brand{display:flex;gap:12px;align-items:center;margin:4px 0 22px}
.dm-mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(145deg,#fff,#dce7ff);color:#17233a!important;
  display:grid;place-items:center;font-weight:800;font-size:20px}
.dm-brand b{display:block;font:700 19px 'Space Grotesk';color:#fff!important}
.dm-brand small{display:block;font-size:11px;color:#8e9caf!important}
.dm-pill{border:1px solid #243249;background:rgba(255,255,255,.035);border-radius:13px;padding:11px;display:flex;gap:10px;align-items:center;margin-top:18px}
.dm-pulse{width:8px;height:8px;border-radius:50%;background:#35d18c;box-shadow:0 0 0 5px rgba(53,209,140,.12)}
.dm-pill small{display:block;font-size:10px;color:#718095!important}
.dm-pill b{display:block;font-size:12px;color:#fff!important}
.dm-label{letter-spacing:.13em;font-size:10px;font-weight:800;color:#8a96a5;text-transform:uppercase}
.dm-hero{border-radius:24px;padding:40px 42px;min-height:330px;color:#fff;position:relative;overflow:hidden;
  background:radial-gradient(circle at 90% 10%,rgba(74,211,226,.18),transparent 26%),
             radial-gradient(circle at 15% 80%,rgba(79,111,255,.25),transparent 34%),linear-gradient(135deg,#0b1422,#16243a);
  box-shadow:0 25px 60px rgba(11,20,34,.18)}
.dm-hero .badge{display:inline-flex;padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.1);font-size:10px;font-weight:800;letter-spacing:.12em;color:#c6d1e2}
.dm-hero h1{font:700 clamp(36px,4vw,56px)/1 'Space Grotesk';letter-spacing:-.045em;margin:24px 0 16px;color:#fff!important}
.dm-hero h1 em{font-style:normal;color:#7f96ff}
.dm-hero p{max-width:560px;color:#aebbd0;font-size:15px;line-height:1.7;margin:0}
.dm-points{display:flex;gap:8px;flex-wrap:wrap;margin-top:26px}
.dm-points span{padding:7px 12px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);
  font-size:12px;color:#dbe4f2}
.dm-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:6px}
.dm-head h2{font:700 28px 'Space Grotesk';margin:4px 0 0}
.dm-status{padding:7px 12px;border-radius:999px;font-size:11px;font-weight:800}
.dm-status.ok{background:var(--green2);color:var(--green)}
.dm-status.wait{background:var(--amber2);color:var(--amber)}
.dm-metrics{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:14px;margin:14px 0}
.dm-card{background:#fff;border:1px solid var(--line);border-radius:17px;padding:20px 22px;box-shadow:0 8px 28px rgba(20,31,51,.03)}
.dm-card strong{display:block;font:700 34px 'Space Grotesk';color:var(--ink);margin-top:6px}
.dm-card small{color:var(--muted);font-size:12px}
.dm-health{display:flex;gap:20px;align-items:center}
.dm-ring{width:104px;height:104px;border-radius:50%;display:grid;place-items:center;flex:none}
.dm-ring div{width:80px;height:80px;border-radius:50%;background:#fff;display:grid;place-items:center;text-align:center}
.dm-ring b{font:700 26px 'Space Grotesk';color:var(--ink);line-height:1}
.dm-ring span{font-size:10px;color:var(--muted)}
.dm-health h3{margin:4px 0;font-size:18px}
.dm-health p{margin:0;color:var(--muted);font-size:13px}
.dm-panels{display:grid;grid-template-columns:1.2fr 1fr;gap:14px;margin-bottom:18px}
.dm-q{display:grid;grid-template-columns:120px 1fr 44px;gap:12px;align-items:center;margin:12px 0}
.dm-q label{font-size:13px;font-weight:600;color:#344054}
.dm-bar{height:7px;border-radius:9px;background:#eef1f5;overflow:hidden;margin:3px 0}
.dm-bar i{display:block;height:100%;border-radius:9px}
.dm-bar.before i{background:#c3cad6}.dm-bar.after i{background:linear-gradient(90deg,#4f6fff,#39c6d8)}
.dm-q small{font-size:10px;color:var(--muted)}
.dm-q strong{font:700 15px 'Space Grotesk';text-align:right}
.dm-act{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #f0f2f6}
.dm-act:last-child{border-bottom:0}
.dm-act .ic{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;font-weight:800;font-size:13px;flex:none}
.dm-act .ok{background:var(--green2);color:var(--green)}.dm-act .warn{background:var(--amber2);color:var(--amber)}
.dm-act .bad{background:var(--red2);color:var(--red)}.dm-act .info{background:var(--blue2);color:var(--blue)}
.dm-act b{display:block;font-size:13px;color:var(--ink)}
.dm-act span{font-size:12px;color:var(--muted)}
.dm-issue{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.dm-issue h4{margin:0;font-size:16px}
.dm-dot{width:9px;height:9px;border-radius:50%;flex:none}
.dm-dot.high{background:var(--red)}.dm-dot.medium{background:var(--amber)}.dm-dot.low{background:var(--blue)}
.dm-tag{font-size:10px;font-weight:800;padding:4px 8px;border-radius:7px;text-transform:uppercase;letter-spacing:.05em}
.dm-tag.high{background:var(--red2);color:var(--red)}.dm-tag.medium{background:var(--amber2);color:var(--amber)}
.dm-tag.low{background:var(--blue2);color:var(--blue)}
.dm-tag.col{background:#f2f4f7;color:#475467;text-transform:none;letter-spacing:0}
.dm-evidence{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0;font-size:13px;color:#344054}
.dm-evidence .dm-label{display:block;margin-bottom:4px}
code{color:var(--blue)!important;background:var(--blue2)!important}
@media (max-width:900px){.dm-metrics,.dm-panels{grid-template-columns:1fr}}
</style>
"""
st.html(CSS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_env():
    """On Streamlit Cloud the API key lives in Secrets; the scripts read it from .env."""
    env_path = os.path.join(HERE, ".env")
    if os.path.exists(env_path):
        return True
    try:
        key = st.secrets["NEBIUS_API_KEY"]
        model = st.secrets.get("NEBIUS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    except Exception:
        return False
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"NEBIUS_API_KEY={key}\nNEBIUS_MODEL={model}\n")
    return True


def run(args):
    """Run one of the DataMedic scripts from the Validation folder and return its output."""
    result = subprocess.run([PY, *args], cwd=HERE, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def read_json(folder, name, default):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [d for d in os.listdir(RESULTS) if os.path.exists(os.path.join(RESULTS, d, "run_info.json"))]
    return sorted(runs, key=lambda d: os.path.getmtime(os.path.join(RESULTS, d)), reverse=True)


def decide(args, folder):
    """Run a human-decision command, then rebuild the PDF and Health Score."""
    ok, out = run(["agent_demo.py", *args, "--dir", folder])
    run(["run_datamedic.py", "--report", folder])
    out = out.replace("Run --pending again before resolving.", "The list below has been updated.")
    lines = [l for l in out.strip().splitlines() if l.strip()]
    failed = not ok or "NOT saved" in out or "rolled back" in out or "not found" in out.lower()
    st.session_state["message"] = ("error" if failed else "success", "\n\n".join(lines) or "Done.")
    st.rerun()


@st.cache_data(show_spinner=False)
def live_issues(cleaned_path, rules_json, mtime):
    """Issues still present in the cleaned file (cached until the file changes)."""
    rules = json.loads(rules_json) if rules_json else None
    return {i["key"]: i for i in profile_dataset(load_data(cleaned_path), rules)["issues"]}


def h(value):
    return escape(str(value))


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

runs = list_runs()
NONE = "—"

# The "Earlier runs" dropdown decides which run is shown ("—" = the start page).
if "pending_pick" in st.session_state:          # a run that just finished
    st.session_state["pick"] = st.session_state.pop("pending_pick")
if st.session_state.get("pick") not in [NONE] + runs:
    st.session_state["pick"] = NONE


def new_analysis():
    st.session_state["pick"] = NONE


with st.sidebar:
    st.html('<div class="dm-brand"><div class="dm-mark">✚</div>'
            '<div><b>DataMedic</b><small>AI data reliability agent</small></div></div>')
    st.button("＋  New analysis", width="stretch", on_click=new_analysis)

    st.html('<div class="dm-label" style="color:#718095;margin-top:18px">Earlier runs</div>')
    if runs:
        st.selectbox("Earlier runs", [NONE] + runs, key="pick", label_visibility="collapsed")
    else:
        st.caption("No runs yet.")

    st.html('<div class="dm-pill"><span class="dm-pulse"></span><div><small>Agent model</small>'
            '<b>NVIDIA Nemotron · Nebius</b></div></div>')


# ---------------------------------------------------------------------------
# Workspace: start a new analysis
# ---------------------------------------------------------------------------

run_name = None if st.session_state["pick"] == NONE else st.session_state["pick"]

if not run_name:
    left, right = st.columns([1.12, 0.88], gap="medium")
    with left:
        st.html("""
<div class="dm-hero">
  <span class="badge">AUTONOMOUS DATA ENGINEERING</span>
  <h1>Turn messy data into<br><em>reliable data.</em></h1>
  <p>DataMedic finds data-quality problems, fixes the safe ones automatically, re-checks every repair,
     and asks you about anything it can't fix with certainty. Your original file is never changed.</p>
  <div class="dm-points"><span>✓ Detect</span><span>✓ Reason</span><span>✓ Repair</span><span>✓ Verify</span></div>
</div>""")
    with right:
        with st.container(border=True, height=330):
            st.html('<div class="dm-label">Dataset</div><h3 style="margin:6px 0 2px">Drop your CSV here</h3>')
            upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
            if upload is not None:
                st.caption(f"{upload.name} · {upload.size / 1024:.1f} KB · ready for analysis")

    with st.container(border=True):
        c1, c2 = st.columns([0.72, 1.28], gap="large")
        c1.html('<div class="dm-label">Dataset context</div>'
                '<h3 style="margin:6px 0">Help the agent understand your data</h3>'
                '<p style="color:#687486;font-size:13px;margin:0">Optional. A one-line description '
                'helps Nemotron make safer decisions.</p>')
        context = c2.text_area("Context", label_visibility="collapsed", height=100,
                               placeholder="Example: Canadian retail orders. Order_ID should be unique, "
                                           "Revenue is in CAD, Province should be a full province name.")

    with st.container(border=True):
        c1, c2 = st.columns([0.72, 1.28], gap="large")
        c1.html('<div class="dm-label">Business rules</div>'
                '<h3 style="margin:6px 0">Rules you want DataMedic to enforce</h3>'
                '<p style="color:#687486;font-size:13px;margin:0">Optional. Leave empty and DataMedic '
                'writes its own rules from the data (unique IDs, no negative amounts, valid ranges).</p>')
        rules_upload = c2.file_uploader("Rules file (JSON)", type=["json"])

    if st.button("✦  Analyze with DataMedic", type="primary", width="stretch"):
        if upload is None:
            st.warning("Choose a CSV file first.")
        elif not ensure_env():
            st.error("No Nebius API key found. Put a .env file next to app.py "
                     "(or add NEBIUS_API_KEY to Secrets on Streamlit Cloud).")
        else:
            os.makedirs(UPLOADS, exist_ok=True)
            data_path = os.path.join(UPLOADS, upload.name)
            with open(data_path, "wb") as f:
                f.write(upload.getbuffer())
            args = ["run_datamedic.py", data_path]
            if rules_upload is not None:
                rules_path = os.path.join(UPLOADS, rules_upload.name)
                with open(rules_path, "wb") as f:
                    f.write(rules_upload.getbuffer())
                args += ["--rules", rules_path]
            if context.strip():
                args += ["--context", context.strip()]

            before = set(runs)
            with st.status("DataMedic is working...", expanded=True) as status:
                st.write("Profiling every column, asking Nemotron for repairs, applying the safe ones "
                         "and re-checking each one. This takes about 20 seconds.")
                ok, out = run(args)
                status.update(label="Analysis complete" if ok else "Something went wrong",
                              state="complete" if ok else "error")
            new = [r for r in list_runs() if r not in before]
            st.session_state["last_log"] = out
            if new:
                st.session_state["pending_pick"] = new[0]
            if not ok:
                st.session_state["message"] = ("error", "The run stopped early. See the run log under Downloads.")
            if new:
                st.rerun()
            else:
                st.code(out, language=None)
    st.stop()


# ---------------------------------------------------------------------------
# Dashboard for one run (every number comes from the run's files)
# ---------------------------------------------------------------------------

folder = os.path.join(RESULTS, run_name)
info = read_json(folder, "run_info.json", {})
pending = read_json(folder, "pending_approvals.json", [])
audit = read_json(folder, "audit_log.json", [])
score = read_json(folder, "health_score.json", {})
profile = read_json(folder, "profile.json", {})
rules = read_json(folder, "rules_used.json", None)
cleaned_path = os.path.join(folder, "cleaned_output.csv")
dataset_name = os.path.basename(info.get("dataset", run_name))
original_severity = {i["key"]: i.get("severity", "low") for i in profile.get("issues", [])}

if "message" in st.session_state:
    kind, text = st.session_state.pop("message")
    (st.success if kind == "success" else st.error)(text)
    # Also pop up in the corner, so it's seen even when scrolled down
    st.toast(text.split("\n\n")[0], icon="✅" if kind == "success" else "⚠️")

status = ('<span class="dm-status ok">● All issues handled</span>' if not pending
          else f'<span class="dm-status wait">● {plural(len(pending), "item")} waiting for you</span>')
st.html(f'<div class="dm-head"><div><div class="dm-label">Analysis · {h(run_name)}</div>'
        f'<h2>{h(dataset_name)}</h2></div>{status}</div>')

repairs = [e for e in audit if e.get("tool") not in NOT_REPAIRS]
auto_fixed = [e for e in repairs if e.get("status") == "applied" and not e.get("approved")]
human_fixed = [e for e in repairs if e.get("status") == "applied" and e.get("approved")]
rolled_back = [e for e in repairs if e.get("status") != "applied"]
decisions = [e for e in audit if e.get("tool") in HUMAN_TOOLS]
summary = profile.get("summary", {})

# ---- Health card + metrics ----
if score.get("after"):
    b, a = score["before"], score["after"]
    after, before = a["overall"], b["overall"]
    ring_color = "#16a874" if after >= 95 else "#4f6fff" if after >= 80 else "#d68a12" if after >= 60 else "#d95757"
    change = after - before
    delta = (f"Up from {before} after verified repairs." if change > 0
             else f"Down from {before}." if change < 0 else f"Unchanged from {before}.")
    health = (f'<div class="dm-card dm-health"><div class="dm-ring" style="background:conic-gradient({ring_color} '
              f'{after * 3.6}deg,#edf0f5 0)"><div><b>{after}</b><span>/100</span></div></div>'
              f'<div><div class="dm-label">Data health</div><h3>{h(a["grade"])}</h3><p>{delta}</p></div></div>')
else:
    health = ('<div class="dm-card"><div class="dm-label">Data health</div><strong>—</strong>'
              '<small>No Health Score for this run</small></div>')
issue_total = summary.get("issue_count", len(profile.get("issues", [])))
st.html(f"""<div class="dm-metrics">{health}
<div class="dm-card"><div class="dm-label">Issues found</div><strong>{issue_total}</strong>
  <small>across {plural(summary.get("columns", 0), "column")}</small></div>
<div class="dm-card"><div class="dm-label">Auto-fixed</div><strong>{len(auto_fixed)}</strong>
  <small>high-confidence, re-checked</small></div>
<div class="dm-card"><div class="dm-label">Needs review</div><strong>{len(pending)}</strong>
  <small>{plural(len(decisions), "decision")} made so far</small></div></div>""")

# ---- Before/after bars + agent activity ----
bars = ""
if score.get("after"):
    for name, label in LABELS.items():
        bd, ad = score["before"]["dimensions"][name], score["after"]["dimensions"][name]
        if bd["score"] is None:
            continue
        bars += (f'<div class="dm-q"><label>{label}</label><div>'
                 f'<div class="dm-bar before"><i style="width:{bd["score"]}%"></i></div>'
                 f'<div class="dm-bar after"><i style="width:{ad["score"]}%"></i></div>'
                 f'<small>Before {bd["score"]} · Now {ad["score"]}</small></div>'
                 f'<strong>{ad["score"]:.0f}</strong></div>')
else:
    bars = '<p style="color:#687486">No Health Score for this run.</p>'

acts = []
for e in repairs:
    tool = e.get("tool", "").replace("_", " ")
    if e.get("status") == "applied":
        effect = (plural(e["rows_removed"], "row") + " removed" if e.get("rows_removed")
                  else plural(e.get("cells_changed", 0), "value") + " changed")
        who = "approved by you" if e.get("approved") else "automatic"
        acts.append(("ok", "✓", f"{tool.capitalize()} · {e.get('column') or 'all columns'}", f"{effect} · {who} · re-checked"))
    else:
        acts.append(("bad", "↺", f"{tool.capitalize()} rolled back", f"{e.get('key')}: the fix didn't solve the issue"))
labels = {"human_resolution": ("info", "✎", "Value entered"), "human_acceptance": ("ok", "✓", "Kept as is"),
          "human_rejection": ("warn", "✕", "Declined")}
for e in decisions:
    cls, icon, text = labels[e["tool"]]
    if e["tool"] == "human_resolution" and e.get("status") != "applied":
        cls, icon, text = "bad", "!", "Entry rejected by the check"
    acts.append((cls, icon, f"{text} · {e.get('column') or ''}", e.get("key", "")))
if pending:
    acts.append(("warn", "!", f"{plural(len(pending), 'issue')} waiting for your decision",
                 "DataMedic never guesses on these"))
activity = "".join(f'<div class="dm-act"><div class="ic {c}">{i}</div><div><b>{h(t)}</b><span>{h(s)}</span></div></div>'
                   for c, i, t, s in acts[-7:]) or '<p style="color:#687486">Nothing yet.</p>'

st.html(f"""<div class="dm-panels">
<div class="dm-card"><div class="dm-label">Quality dimensions</div><h3 style="margin:6px 0 4px">Before vs. now</h3>{bars}</div>
<div class="dm-card"><div class="dm-label">Agent activity</div><h3 style="margin:6px 0 4px">What DataMedic did</h3>{activity}</div>
</div>""")

tab_decide, tab_fixed, tab_data, tab_files = st.tabs(
    [f"Needs your decision ({len(pending)})", "Audit trail", "Cleaned data", "Downloads"])

# ---- Tab 1: human decisions ----
with tab_decide:
    if not pending:
        st.success("Nothing is waiting. Every issue was repaired, resolved or reviewed.")
    else:
        issues = live_issues(cleaned_path, json.dumps(rules) if rules else "",
                             os.path.getmtime(cleaned_path)) if os.path.exists(cleaned_path) else {}
        df = load_data(cleaned_path) if os.path.exists(cleaned_path) else None

        filt = st.segmented_control("Severity", ["All", "High", "Medium", "Low"], default="All",
                                    label_visibility="collapsed")
        order = {"high": 0, "medium": 1, "low": 2}
        shown = sorted(pending, key=lambda p: order.get(
            (issues.get(p["key"]) or {}).get("severity", original_severity.get(p["key"], "low")), 3))

        for p in shown:
            iid, key, tool = p["issue_id"], p["key"], p["tool"]
            live = issues.get(key)
            sev = (live or {}).get("severity", original_severity.get(key, "low"))
            if filt and filt != "All" and sev != filt.lower():
                continue
            has_fix = tool != "flag_for_review"
            issue_type = key.split(":")[1] if ":" in key else key
            column = (live or {}).get("column") or p.get("params", {}).get("column") or "*"

            with st.container(border=True):
                st.html(f'<div class="dm-issue"><span class="dm-dot {sev}"></span>'
                        f'<h4>{h(issue_type.replace("_", " ").capitalize())}</h4>'
                        f'<span class="dm-tag col">{h("all columns" if column == "*" else column)}</span>'
                        f'<span class="dm-tag {sev}">{sev}</span>'
                        f'<span style="color:#687486;font-size:12px;margin-left:auto">{iid}'
                        f'{" · " + plural(live["row_count"], "row") if live else ""}</span></div>'
                        f'<div class="dm-evidence"><span class="dm-label">Evidence</span>'
                        f'{h(live.get("evidence", "")) if live else "No longer detected."}</div>'
                        f'<div class="dm-evidence"><span class="dm-label">Nemotron\'s proposal · '
                        f'confidence {float(p.get("confidence", 0)):.0%}</span>'
                        f'<code>{h(tool)}</code> {h(p.get("rationale", ""))}</div>')
                if has_fix and p.get("params", {}).get("mapping"):
                    with st.expander("See the proposed mapping"):
                        st.json(p["params"]["mapping"])
                if live and df is not None and live.get("rows"):
                    with st.expander("See the affected rows"):
                        st.dataframe(df.iloc[live["rows"][:200]], width="stretch")

                st.text_input("Reason", key=f"reason_{iid}", label_visibility="collapsed",
                              placeholder="Reason, if you decline (optional)")
                b1, b2, b3 = st.columns(3)
                if has_fix and b1.button("Approve fix", key=f"ap_{iid}", type="primary", width="stretch"):
                    decide(["--approve", "--issue", iid], folder)
                if b2.button("Keep as is", key=f"ac_{iid}", width="stretch",
                             help="The data is fine as it is (e.g. a real large value)."):
                    decide(["--accept", "--issue", iid], folder)
                if b3.button("Decline", key=f"rj_{iid}", width="stretch",
                             help="Say no to this proposal. The data is not changed and the issue stays open."):
                    decide(["--reject", "--issue", iid, "--reason",
                            st.session_state.get(f"reason_{iid}") or "Declined in the web app."], folder)

                if live and live.get("rows") and column != "*":
                    with st.expander("Enter the correct value for a row"):
                        with st.form(f"resolve_{iid}", border=False):
                            r1, r2, r3 = st.columns([1, 2, 1], vertical_alignment="bottom")
                            row = r1.selectbox("Row", live["rows"], key=f"row_{iid}")
                            value = r2.text_input(f"New value for {column}", key=f"val_{iid}",
                                                  help="Use the standard format: dates as YYYY-MM-DD, plain numbers.")
                            if r3.form_submit_button("Save", width="stretch") and value.strip():
                                decide(["--resolve", "--issue", iid, "--row", str(row),
                                        "--value", value.strip()], folder)
                        if df is not None:
                            st.caption(f"Current value in row {row}: '{df.iloc[row][column]}'")

# ---- Tab 2: audit trail ----
with tab_fixed:
    if repairs:
        st.subheader("Repairs")
        st.dataframe(pd.DataFrame([{
            "Issue": e.get("key"), "Tool": e.get("tool"),
            "Approved by": "Human" if e.get("approved") else "Automatic",
            "Effect": (plural(e["rows_removed"], "row") + " removed" if e.get("rows_removed")
                       else plural(e.get("cells_changed", 0), "cell")),
            "Result": "Validated" if e.get("status") == "applied" else "Rolled back",
        } for e in repairs]), hide_index=True, width="stretch")

    if decisions:
        st.subheader("Human decisions")
        label = {"human_resolution": "Entered a value", "human_acceptance": "Kept as is",
                 "human_rejection": "Declined"}
        st.dataframe(pd.DataFrame([{
            "Issue": e.get("key"), "Decision": label[e["tool"]],
            "Row": e.get("row"), "Old value": e.get("old_value"), "New value": e.get("new_value"),
            "Reason": e.get("rationale") if e["tool"] == "human_rejection" else None,
            "Result": {"applied": "Saved", "rejected": "Not saved (failed the check)"}.get(
                e.get("status"), (e.get("status") or "").replace("_", " ")),
        } for e in decisions]), hide_index=True, width="stretch")

    changes = [dict(c, Issue=e.get("key"), By="Human" if e.get("approved") else "Automatic")
               for e in repairs if e.get("status") == "applied" for c in e.get("changes", [])]
    if changes:
        st.subheader(f"Every changed value ({len(changes)})")
        st.dataframe(pd.DataFrame(changes).rename(columns={
            "row": "Row", "column": "Column", "old_value": "Old value", "new_value": "New value"}),
            hide_index=True, width="stretch")
    if not (repairs or decisions):
        st.info("Nothing has been changed yet.")

# ---- Tab 3: cleaned data ----
with tab_data:
    if os.path.exists(cleaned_path):
        st.dataframe(load_data(cleaned_path), width="stretch", height=500)
    else:
        st.info("No cleaned file was created for this run.")

# ---- Tab 4: downloads ----
with tab_files:
    base = os.path.splitext(dataset_name)[0]
    cols = st.columns(3)
    for col, (name, label, mime, out) in zip(cols, [
        ("audit_report.pdf", "Audit report (PDF)", "application/pdf", f"{base}_audit_report.pdf"),
        ("cleaned_output.csv", "Cleaned data (CSV)", "text/csv", f"{base}_cleaned.csv"),
        ("audit_log.json", "Audit log (JSON)", "application/json", f"{base}_audit_log.json"),
    ]):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                col.download_button(label, f.read(), file_name=out, mime=mime, key=f"dl_{name}", width="stretch")
    if info.get("context"):
        st.caption(f"Dataset context given to Nemotron: {info['context']}")
    if st.session_state.get("last_log"):
        with st.expander("Run log"):
            st.code(st.session_state["last_log"], language=None)
