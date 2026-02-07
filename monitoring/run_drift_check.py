import pandas as pd
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Paths
REFERENCE_PATH = Path("data/reference/reference.csv")
CURRENT_PATH = Path("data/current/current.csv")
REPORT_PATH = Path("reports/drift_report.html")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load data
reference = pd.read_csv(REFERENCE_PATH)
current = pd.read_csv(CURRENT_PATH)

# Drift report
report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html(str(REPORT_PATH))

print(" Drift report generated at:", REPORT_PATH)
