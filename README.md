# DataMedic

**AI-powered data reliability agent built for the Nebius × NVIDIA Global AI Hackathon.**

DataMedic detects data-quality problems, reasons about how they should be handled, automatically repairs high-confidence issues, validates every repair, and sends uncertain cases to a human instead of guessing.

The application uses **NVIDIA Nemotron through Nebius** for agent reasoning and a deterministic Python repair and validation layer for safe execution.

---

## What DataMedic Does

DataMedic takes a CSV dataset and runs an end-to-end data reliability workflow:

1. **Detect** data-quality problems.
2. **Reason** about each issue using NVIDIA Nemotron.
3. **Repair** high-confidence issues automatically.
4. **Verify** every automatic repair by profiling the dataset again.
5. **Rollback** repairs that fail validation.
6. **Request human review** when a correction cannot be made safely.
7. **Audit** automatic and human changes.
8. **Export** the cleaned dataset and audit artifacts.

The original uploaded dataset is never modified.

---

## Key Features

### Data Profiling

DataMedic detects issues including:

- Missing values
- Duplicate rows
- Duplicate identifiers
- Category inconsistencies
- Extra whitespace
- Currency formatting inconsistencies
- Mixed date formats
- Ambiguous dates
- Future dates
- Negative numeric values
- Statistical outliers
- Business-rule violations

### Nemotron Reasoning

Detected issues are passed to NVIDIA Nemotron, which proposes an action and confidence score.

DataMedic does not allow the model to directly modify the dataset. Model proposals must pass through the repair and validation layer.

### Safe Automatic Repair

High-confidence repairs can be executed automatically using deterministic repair tools.

Examples include:

- Category standardization
- Currency parsing
- Whitespace trimming
- Date standardization
- Duplicate-row removal

After a repair, DataMedic profiles the dataset again to verify that the intended issue was actually resolved.

If validation fails, the repair is rolled back.

### Human-in-the-Loop Review

Issues that cannot be corrected safely are sent to the user.

The reviewer can:

- Keep the value as-is
- Decline a proposed action
- Enter a corrected value for a specific row
- Approve supported repair actions

DataMedic records these decisions in the audit trail.

### Business Rules

Users can optionally upload a JSON rules file.

Supported rules include constraints such as:

- Unique identifiers
- Minimum and maximum numeric values
- Allowed values
- Required values
- Canadian province validation
- Pattern validation

If no rules file is supplied, DataMedic can infer conservative rules from the dataset.

### Dataset Context

Users can optionally provide a natural-language description of the dataset to give Nemotron additional business context when reasoning about detected issues.

### Data Health Score

DataMedic calculates a data-health score across quality dimensions including:

- Completeness
- Uniqueness
- Validity
- Consistency
- Business-rule compliance

The interface shows data quality before and after verified repairs.

### Full Auditability

DataMedic records:

- Detected issue
- Proposed repair
- Confidence
- Repair tool
- Approval source
- Old value
- New value
- Affected row
- Validation result
- Rollbacks
- Human decisions

Outputs include:

- Cleaned CSV
- Audit log JSON
- Audit report PDF

---

## Safety Model

DataMedic follows a simple principle:

> **Automate when the evidence is strong. Ask when it isn't.**

The LLM acts as the reasoning layer, while deterministic Python tools perform actual data modifications.

Automatic repairs are re-checked after execution. Failed repairs are rolled back rather than silently accepted.

For ambiguous or unsupported corrections, DataMedic requests human review instead of inventing data.

---

## Project Structure

```text
.
├── .streamlit/
│   └── config.toml
├── agent_demo.py
├── app.py
├── audit_to_pdf.py
├── check_coverage.py
├── employee_rules.json
├── generate_dataset.py
├── health_score.py
├── profiler.py
├── repair_tools.py
├── requirements.txt
├── rules.json
├── run_datamedic.py
└── titanic_rules.json
```

### Main Components

**`profiler.py`**  
Profiles datasets, detects data-quality issues, evaluates business rules, and supports automatic rule inference.

**`agent_demo.py`**  
Connects detected issues to NVIDIA Nemotron and manages agent reasoning and repair proposals.

**`repair_tools.py`**  
Contains deterministic repair operations and repair validation/rollback logic.

**`health_score.py`**  
Calculates DataMedic's data-health metrics.

**`audit_to_pdf.py`**  
Generates the downloadable PDF audit report.

**`run_datamedic.py`**  
Runs the DataMedic workflow used by the Streamlit interface.

**`generate_dataset.py`**  
Generates synthetic messy data for testing DataMedic's detection and repair workflow.

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Dependencies:

- pandas
- openai
- reportlab
- streamlit

---

## Configuration

DataMedic requires credentials for the Nemotron endpoint.

Store credentials securely using environment variables or Streamlit secrets.

**Never commit API keys or `.env` files to the repository.**

The repository's `.gitignore` excludes:

```text
.env
results/
uploads/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

---

## Running DataMedic

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then:

1. Upload a CSV.
2. Optionally describe the dataset.
3. Optionally upload business rules as JSON.
4. Run DataMedic.
5. Review uncertain issues.
6. Inspect the audit trail.
7. Download the cleaned dataset and audit artifacts.

---

## Tested Workflows

DataMedic has been tested with both synthetic and external datasets.

Testing covered:

- Automatic issue detection
- Automatic repairs
- Post-repair validation
- Failed-repair rollback
- Human review
- Manual row corrections
- Keep-as-is decisions
- Duplicate resolution
- Automatic rule inference
- Explicit JSON business rules
- Natural-language dataset context
- Empty and invalid CSV handling
- Cleaned CSV generation
- JSON audit generation
- PDF audit report generation
- Fresh-session and new-analysis workflows

---

## Technology

- **NVIDIA Nemotron** — agent reasoning
- **Nebius** — model infrastructure/API
- **Python** — profiling, validation and repair execution
- **pandas** — data processing
- **Streamlit** — interactive web application
- **ReportLab** — PDF audit reporting

---

## Team & Contributions

### Primary Contributors

- **Jason Thai** — Project Lead, Reasoning Lead, Profiling, Validation & Health Score Engineer, Frontend Engineer, Product & Pitch
- **Ria Chahar** — Agent Architect, Repair Tools Engineer, Backend & Infrastructure Engineer, Product & Pitch

### Testing & QA

- **Omer Irfan** — Application Testing & QA Feedback

### Additional Team Members

- **Prabal Pratap Sharma**
- **Nashwa Meher Ahmed Lnu**
- **Shreya Vinod Gupta**
- **Jesika Dasadia**

---

## Hackathon

Built for the **Nebius × NVIDIA Global AI Hackathon**.

DataMedic demonstrates an auditable, human-in-the-loop approach to AI-assisted data engineering: allowing an AI agent to reason about messy data while keeping actual modifications deterministic, validated, reversible, and reviewable.
