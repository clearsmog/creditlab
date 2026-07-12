"""ML challenger: gradient boosting vs the WoE scorecard champion.

The champion/challenger question banks actually ask: how much discrimination
does the interpretable scorecard leave on the table? Trees take the raw
ratios (no WoE needed — splits are invariant to monotone transforms) and
handle missing values natively. Explainability comes from exact TreeSHAP,
which xgboost computes itself (predict with pred_contribs=True).

Run the comparison:  uv run python -m creditlab.models.challenger
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from creditlab.models.metrics import ks_stat, psi, roc_auc
from creditlab.models.scorecard import RATIO_FEATURES, Scorecard

PARAMS = {
    # shallow, slow-learning trees: ~8k rows and 500 defaults do not support
    # deep interactions, and the OOT gap is the thing to protect
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "eval_metric": "auc",
}


def fit_challenger(train: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(train[RATIO_FEATURES], y)
    return model


def shap_importance(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature (exact TreeSHAP, computed by xgboost natively)."""
    booster = model.get_booster()
    dmat = xgb.DMatrix(df[RATIO_FEATURES], missing=np.nan)
    contribs = booster.predict(dmat, pred_contribs=True)  # last column = bias
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    return pd.Series(mean_abs, index=RATIO_FEATURES).sort_values(ascending=False)


def main() -> None:
    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train, test = df[df["fyear"] <= 2019], df[df["fyear"] >= 2020]
    y_tr, y_te = train["default_within_1y"], test["default_within_1y"]

    card = Scorecard().fit(train, y_tr)
    xgbm = fit_challenger(train, y_tr)

    print(f"{'model':24s} {'train AUC':>9s} {'test AUC':>9s} {'test KS':>8s}")
    for name, predict in [
        ("champion scorecard", card.predict_pd),
        ("challenger xgboost", lambda d: xgbm.predict_proba(d[RATIO_FEATURES])[:, 1]),
    ]:
        p_tr, p_te = predict(train), predict(test)
        print(
            f"{name:24s} {roc_auc(y_tr, p_tr):9.3f} {roc_auc(y_te, p_te):9.3f} "
            f"{ks_stat(y_te, p_te):8.3f}"
        )

    print("\nchallenger TreeSHAP importance (top 8, on test):")
    print(shap_importance(xgbm, test).head(8).to_string(float_format=lambda x: f"{x:.3f}"))

    # population stability of the champion score between windows
    stability = psi(card.predict_pd(train), card.predict_pd(test))
    print(f"\nchampion PD PSI train->test: {stability:.3f} "
          "(<0.10 stable, 0.10-0.25 monitor, >0.25 shifted)")


if __name__ == "__main__":
    main()
