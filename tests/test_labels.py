"""Episode collapsing for 8-K Item 1.03 default events."""

import pandas as pd
import pytest

import creditlab.data.labels as labels


def _patch_events(monkeypatch, dates):
    events = pd.DataFrame({
        "filingDate": pd.to_datetime(dates),
        "accessionNumber": [f"acc-{i}" for i in range(len(dates))],
        "items": ["1.03"] * len(dates),
    })
    monkeypatch.setattr(labels, "bankruptcy_events", lambda cik: events)


def test_entry_and_emergence_collapse_to_one_episode(monkeypatch):
    _patch_events(monkeypatch, ["2020-05-26", "2021-06-16"])  # Hertz pattern
    assert labels.default_episodes(1) == [pd.Timestamp("2020-05-26")]


def test_long_chapter11_does_not_split(monkeypatch):
    # gaps measured from the LAST event, so a >2y case chained by interim
    # filings stays one episode
    _patch_events(monkeypatch, ["2019-01-10", "2020-06-01", "2021-11-15"])
    assert labels.default_episodes(1) == [pd.Timestamp("2019-01-10")]


def test_separate_episodes_detected(monkeypatch):
    _patch_events(monkeypatch, ["2010-03-01", "2016-09-01"])
    assert labels.default_episodes(1) == [
        pd.Timestamp("2010-03-01"), pd.Timestamp("2016-09-01"),
    ]


def test_no_events(monkeypatch):
    _patch_events(monkeypatch, [])
    assert labels.default_episodes(1) == []
