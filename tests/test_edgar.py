"""Annual-record extraction on a synthetic companyfacts fixture (no network)."""

import numpy as np
import pandas as pd

from creditlab.data.edgar import annual_record


def _obs(end, val, start=None, filed="2020-03-01", fp="FY", form="10-K"):
    d = {"end": end, "val": val, "fy": 2019, "fp": fp, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


FACTS = {
    "entityName": "TestCo",
    "facts": {
        "us-gaap": {
            "Assets": {"units": {"USD": [
                _obs("2018-12-31", 90.0),
                _obs("2019-06-30", 55.0),          # quarterly snapshot tagged FY
                _obs("2019-12-31", 100.0),
                _obs("2019-12-31", 101.0, filed="2021-02-01"),  # restatement
            ]}},
            "Revenues": {"units": {"USD": [
                _obs("2018-12-31", 50.0, start="2018-01-01"),
                _obs("2019-12-31", 60.0, start="2019-01-01"),
                _obs("2019-12-31", 30.0, start="2019-07-01"),   # H2 stub period
            ]}},
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                # cover-page date ~6 weeks after FY end, no fp/form filter match
                {"end": "2020-02-14", "val": 10.0, "filed": "2020-02-14"},
            ]}},
        },
    },
}


FY18, FY19 = pd.Timestamp("2018-12-31"), pd.Timestamp("2019-12-31")


def test_annual_record_shape_and_values():
    df = annual_record(FACTS)
    # quarterly snapshot (2019-06-30) has no full-year flow -> dropped
    assert list(df.index) == [FY18, FY19]
    # latest-filed value wins for restated periods
    assert df.loc[FY19, "assets"] == 101.0
    # stub-period (H2) revenue observation is excluded by the duration window
    assert df.loc[FY19, "revenue"] == 60.0


def test_shares_nearest_date_fallback():
    df = annual_record(FACTS)
    # 2020-02-14 cover date is within 180d of FY2019 end...
    assert df.loc[FY19, "shares_outstanding"] == 10.0
    # ...but not of FY2018 end
    assert np.isnan(df.loc[FY18, "shares_outstanding"])
