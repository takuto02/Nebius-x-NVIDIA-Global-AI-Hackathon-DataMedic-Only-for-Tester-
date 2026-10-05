"""
DataMedic audit report (PDF)
----------------------------
Turns a run's audit log into a readable report:

    1. Summary          - repairs applied, cells changed, human decisions, rollbacks, pending
    2. Repairs          - every repair action and whether it was validated or rolled back
    3. Cell changes     - row by row: old value -> new value, and who made it (auto or human)
    4. Human-accepted   - issues a human reviewed and kept as they are (or declined)
    5. Pending review   - issues still waiting for a human (never guessed)

Setup (once):
    pip install reportlab

Usage:
    python audit_to_pdf.py --dir results/my_run                      # reads the run's files
    python audit_to_pdf.py --dir results/my_run --out report.pdf --dataset messy_retail.csv
    python audit_to_pdf.py audit_log.json --pending pending_approvals.json
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#59636e")
HEADER_BG = colors.HexColor("#0b3d2e")
ZEBRA = colors.HexColor("#f4f6f8")
RULE = colors.HexColor("#d0d7de")
GREEN = colors.HexColor("#1b7f3b")
RED = colors.HexColor("#b3261e")
AMBER = colors.HexColor("#9a6700")

MAX_CELL_ROWS = 300  # keep the PDF readable; the JSON log has everything


def load_json(path, default):
    if not path or not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate {"entries": [...]} style files
        lists = [v for v in data.values() if isinstance(v, list)]
        return max(lists, key=len) if lists else default
    return data


def show(value):
    if value is None:
        return "—"
    s = str(value)
    if s == "":
        return "(blank)"
    if s != s.strip():
        s = s.replace(" ", "·")  # make hidden spaces visible
    return s


def build(audit, pending, out_path, dataset):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=20, textColor=INK, alignment=0, spaceAfter=4)
    sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9.5, textColor=MUTED, leading=13)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, textColor=INK, spaceBefore=12, spaceAfter=6)
    cell = ParagraphStyle("c", parent=styles["Normal"], fontSize=8, leading=10, textColor=INK)
    head = ParagraphStyle("h", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=9, leading=12.5, textColor=INK)

    repairs = [e for e in audit if e.get("tool") not in ("human_resolution", "human_acceptance",
                                                           "human_rejection", "resolved_by_other_fix")]
    side_resolved = [e for e in audit if e.get("tool") == "resolved_by_other_fix"]
    applied = [e for e in repairs if e.get("status") == "applied"]
    rolled = [e for e in repairs if e.get("status") != "applied"]
    human_res = [e for e in audit if e.get("tool") == "human_resolution" and e.get("status") == "applied"]
    human_rej = [e for e in audit if e.get("tool") == "human_resolution" and e.get("status") != "applied"]
    accepted = [e for e in audit if e.get("status") == "accepted_as_is"]
    declined = [e for e in audit if e.get("status") == "rejected_by_human"]

    cell_rows = []
    for e in applied:
        for c in e.get("changes", []):
            cell_rows.append((c.get("row"), c.get("column"), c.get("old_value"), c.get("new_value"),
                              e.get("key"), "Approved by human" if e.get("approved") else "Automatic"))
    for e in human_res:
        cell_rows.append((e.get("row"), e.get("column"), e.get("old_value"), e.get("new_value"),
                          e.get("key"), "Entered by human"))
    cells_changed = sum(e.get("cells_changed", 0) for e in applied) + len(human_res)
    rows_removed = sum(e.get("rows_removed", 0) for e in applied)

    story = [Paragraph("DataMedic Audit Report", title)]
    meta = f"Generated {datetime.now():%B %d, %Y at %H:%M}"
    if dataset:
        meta += f"  ·  Dataset: {escape(dataset)}"
    story += [Paragraph(meta, sub), Spacer(1, 8)]

    # ---- 1. Summary ----
    story.append(Paragraph("Summary", h2))
    tiles = [["Repairs applied", "Cells changed", "Rows removed", "Human decisions",
              "Rolled back", "Pending review"],
             [str(len(applied)), str(cells_changed), str(rows_removed),
              str(len(human_res) + len(accepted) + len(declined)), str(len(rolled)), str(len(pending))]]
    t = Table(tiles, colWidths=[1.66 * inch] * 6)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica", 8.5), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 18), ("TEXTCOLOR", (0, 1), (-1, 1), INK),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.6, RULE),
        ("BACKGROUND", (0, 0), (-1, -1), ZEBRA), ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
    ]))
    story += [t, Spacer(1, 8), Paragraph(
        "DataMedic applies repairs only where the evidence is sufficient, re-checks every repair and "
        "rolls back any that don't work, sends uncertain cases to a human, and leaves values it cannot "
        "infer safely unresolved instead of inventing them. Every entry below was recorded by the code "
        "that made the change.", body)]

    def table(header, rows, widths, status_col=None):
        data = [[Paragraph(h, head) for h in header]]
        cmds = [("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        for i, r in enumerate(rows, start=1):
            cells = []
            for j, v in enumerate(r):
                text = escape(show(v))
                if j == status_col:
                    colour = GREEN if v in ("Validated", "Accepted as-is") else RED if "Rolled" in str(v) or "Rejected" in str(v) else AMBER
                    text = f'<font color="{colour.hexval()}"><b>{text}</b></font>'
                cells.append(Paragraph(text, cell))
            data.append(cells)
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
        tbl = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
        tbl.setStyle(TableStyle(cmds))
        return tbl

    # ---- 2. Repairs ----
    story.append(Paragraph("Repairs", h2))
    if repairs:
        rows = []
        for e in repairs:
            effect = (f"{e['rows_removed']} rows removed" if e.get("rows_removed")
                      else f"{e.get('cells_changed', 0)} cells" if e.get("status") == "applied" else "no change kept")
            result = "Validated" if e.get("status") == "applied" and e.get("validated") else "Rolled back"
            rows.append((e.get("key"), e.get("tool"), e.get("column"),
                         "Human" if e.get("approved") else "Automatic",
                         effect, f"{float(e['confidence']):.0%}" if e.get("confidence") is not None else None,
                         result))
        story.append(table(["Issue", "Tool", "Column", "Approved by", "Effect", "Confidence", "Result"],
                           rows, [2.6, 1.5, 1.0, 0.9, 1.3, 0.8, 0.9], status_col=6))
    else:
        story.append(Paragraph("No repairs were applied.", body))

    # ---- 3. Cell-level changes ----
    story.append(Paragraph("Every changed value", h2))
    if cell_rows:
        cell_rows.sort(key=lambda r: (r[0] if isinstance(r[0], int) else 10**9, str(r[1])))
        shown = cell_rows[:MAX_CELL_ROWS]
        story.append(table(["Row", "Column", "Old value", "New value", "Issue", "Made by"],
                           shown, [0.5, 1.1, 1.8, 1.8, 2.8, 1.2]))
        if len(cell_rows) > MAX_CELL_ROWS:
            story.append(Paragraph(f"Showing the first {MAX_CELL_ROWS} of {len(cell_rows)} changed values. "
                                   f"The full list is in the audit log (JSON).", sub))
        story.append(Paragraph("Row numbers start at 0. A '·' marks a space at the start or end of a value.", sub))
    else:
        story.append(Paragraph("No individual values were changed.", body))

    if human_rej:
        story.append(Paragraph("Human entries that failed the check (not saved)", h2))
        story.append(table(["Row", "Column", "Entered value", "Issue", "Result"],
                           [(e.get("row"), e.get("column"), e.get("new_value"), e.get("key"), "Rejected")
                            for e in human_rej], [0.6, 1.2, 2.0, 4.2, 1.0], status_col=4))

    # ---- 4. Human-accepted ----
    story.append(Paragraph("Reviewed by a human and kept as-is", h2))
    if accepted:
        story.append(table(["Issue", "Column", "Rows affected", "Decision"],
                           [(e.get("key"), e.get("column"), e.get("rows_affected"), "Accepted as-is")
                            for e in accepted], [4.4, 1.4, 1.4, 1.8], status_col=3))
    else:
        story.append(Paragraph("No issues were accepted as-is.", body))

    if declined:
        story.append(Paragraph("Declined by a human (data not changed, issue still open)", h2))
        story.append(table(["Issue", "Declined fix", "Rows affected", "Reason", "Decision"],
                           [(e.get("key"), e.get("rejected_tool"), e.get("rows_affected"), e.get("rationale"),
                             "Rejected") for e in declined], [2.8, 1.4, 1.0, 3.8, 1.0], status_col=4))

    if side_resolved:
        story.append(Paragraph("Resolved as a side effect of another fix", h2))
        story.append(table(["Issue", "Column", "Why"],
                           [(e.get("key"), e.get("column"), e.get("rationale")) for e in side_resolved],
                           [3.6, 1.4, 5.0]))

    # ---- 5. Pending ----
    story.append(Paragraph("Still waiting for a human decision", h2))
    if pending:
        story.append(table(["Issue", "Column", "Proposed", "Why it needs a human"],
                           [(p.get("key"), p.get("params", {}).get("column"), p.get("tool"), p.get("rationale"))
                            for p in pending], [2.8, 1.1, 1.4, 4.7]))
    else:
        story.append(Paragraph("Nothing is waiting: every issue was repaired, resolved or reviewed.", body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.5 * inch, 0.4 * inch, "DataMedic · Nebius x NVIDIA Global AI Hackathon")
        canvas.drawRightString(10.5 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=landscape(letter), leftMargin=0.5 * inch,
                            rightMargin=0.5 * inch, topMargin=0.5 * inch, bottomMargin=0.65 * inch,
                            title="DataMedic Audit Report", author="DataMedic")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return {"repairs": len(applied), "cells": cells_changed, "human": len(human_res) + len(accepted) + len(declined),
            "rolled_back": len(rolled), "pending": len(pending)}


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


if __name__ == "__main__":
    folder = arg("--dir")
    positional = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    if not folder and not positional:
        print(__doc__)
        sys.exit(1)
    audit_path = positional or os.path.join(folder, "audit_log.json")
    pending_path = arg("--pending") or (os.path.join(folder, "pending_approvals.json") if folder else None)
    out = arg("--out") or (os.path.join(folder, "audit_report.pdf") if folder else audit_path.rsplit(".", 1)[0] + ".pdf")

    audit = load_json(audit_path, [])
    pending = load_json(pending_path, [])
    stats = build(audit, pending, out, arg("--dataset"))
    print(f"Saved {out}")
    print(f"  {stats['repairs']} repairs, {stats['cells']} values changed, {stats['human']} human decisions, "
          f"{stats['rolled_back']} rolled back, {stats['pending']} pending")
