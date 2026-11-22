import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUNS = Path(__file__).resolve().parents[1] / "experiments" / "runs"


@pytest.mark.skipif(not any(RUNS.glob("FD001-*.json")), reason="no experiment records present")
def test_report_is_built_from_registry_records_and_has_every_section():
    sys.path.insert(0, str(SCRIPTS))
    import make_report

    text = make_report.build()
    for heading in ("## 1. Model leaderboards", "## 2. Ablations", "## 4. Failure warnings", "## 5. Prediction intervals",
                    "## 6. Anomaly detection", "## 7. Explainability", "## 9. Reproducibility"):
        assert heading in text
    assert "lightgbm_window50" in text and "constant" in text
    assert "synthetic" in text.lower()


def test_overlap_and_formatting_helpers():
    sys.path.insert(0, str(SCRIPTS))
    import make_report

    assert make_report.overlaps((1, 3), (2, 5)) and not make_report.overlaps((1, 2), (3, 4))
    assert make_report.f(float("nan")) == "n/a" and make_report.f(None) == "n/a" and make_report.f(1.234) == "1.23"
