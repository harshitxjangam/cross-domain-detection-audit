# Submission report

The checked-in PDF and figures are the submission deliverables. To regenerate
them after recovering the ignored experiment artifacts into
`artifacts/final_evidence/`, create a local reporting environment and run:

```bash
python3 -m venv .report-venv
.report-venv/bin/python -m pip install matplotlib reportlab pypdf pillow
.report-venv/bin/python scripts/analysis/build_submission.py
```

The script produces the PDF, four data-derived figures, and the compact results
tables. It does not use copied benchmark values or download any data.
