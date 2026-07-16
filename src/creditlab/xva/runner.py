"""Run ORE on generated configs and parse exposure/XVA reports.

Requires the optional ``xva`` dependency group:

  uv sync --extra xva      # installs open-source-risk-engine
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pandas as pd

from creditlab.xva.configs import XvaInputs, write_all


@dataclass
class XvaResults:
    inputs: XvaInputs
    exposure: pd.DataFrame      # netting-set profile: Time, EPE, ENE, PFE
    xva: pd.DataFrame           # ORE xva report (netting-set + per-trade rows)
    npv: pd.DataFrame           # T0 NPV per trade
    work_dir: str               # inputs + raw ORE reports for inspection

    @property
    def cva(self) -> float:
        """Netting-set CVA (the row without a trade id)."""
        ns_row = self.xva[self.xva["#TradeId"].isna()]
        return float(ns_row["CVA"].iloc[0])

    @property
    def peak_epe(self) -> float:
        return float(self.exposure["EPE"].max())

    @property
    def peak_pfe(self) -> float:
        return float(self.exposure["PFE"].max())


def run_xva(inputs: XvaInputs, work_dir: str | None = None) -> XvaResults:
    """Generate ORE inputs, run simulation + XVA, return parsed results.

    ORE swallows analytic errors into its log, so success is checked by the
    presence of the xva report; failures raise with the first logged ALERT.
    """
    try:
        from ORE import OREApp, Parameters
    except ImportError as e:
        raise ImportError(
            "ORE Python bindings not installed - run: uv sync --extra xva"
        ) from e

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="creditlab-xva-")
    work_dir = os.path.abspath(work_dir)
    write_all(inputs, work_dir)

    params = Parameters()
    params.fromFile(os.path.join(work_dir, "ore.xml"))
    app = OREApp(params, False)
    app.run()

    out = os.path.join(work_dir, "Output")
    xva_csv = os.path.join(out, "xva.csv")
    if not os.path.exists(xva_csv):
        raise RuntimeError(_first_alert(os.path.join(out, "log.txt")))

    exposure = pd.read_csv(
        os.path.join(out, f"exposure_nettingset_{inputs.counterparty}.csv")
    )
    return XvaResults(
        inputs=inputs,
        exposure=exposure,
        xva=pd.read_csv(xva_csv),
        npv=pd.read_csv(os.path.join(out, "npv.csv")),
        work_dir=work_dir,
    )


def _first_alert(log_path: str) -> str:
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if line.startswith("ALERT"):
                    return f"ORE run failed: {line.strip()}"
    return f"ORE run failed: no xva report produced (see {log_path})"
