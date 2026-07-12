"""PD scorecard: WoE features -> logistic regression -> points scale.

Standard development recipe:
  1. bin every candidate ratio (monotonic WoE), rank by Information Value
  2. keep IV >= threshold, drop the weaker of highly correlated pairs
  3. logistic regression on WoE features (statsmodels: p-values are part of
     the model documentation banks require)
  4. map log-odds to a points scale via PDO ("points to double the odds")

Run the out-of-time validation:  uv run python -m creditlab.models.scorecard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from creditlab.models.binning import WoEBinner
from creditlab.models.metrics import accuracy_ratio, gini, ks_stat, roc_auc

RATIO_FEATURES = [
    "leverage", "debt_to_equity", "ltd_to_assets",
    "interest_coverage", "cfo_to_debt",
    "current_ratio", "cash_to_assets", "wc_to_assets",
    "roa", "operating_margin", "re_to_assets",
    "asset_turnover", "log_assets",
]


class Scorecard:
    def __init__(self, min_iv: float = 0.05, max_corr: float = 0.8,
                 pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0):
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.pdo, self.base_score, self.base_odds = pdo, base_score, base_odds
        self.binners_: dict[str, WoEBinner] = {}
        self.features_: list[str] = []
        self.model_ = None

    def fit(self, df: pd.DataFrame, y: pd.Series, features: list[str] = RATIO_FEATURES) -> "Scorecard":
        # 1. bin and rank by IV
        binners = {f: WoEBinner().fit(df[f], y) for f in features}
        ranked = sorted(features, key=lambda f: binners[f].iv_, reverse=True)
        candidates = [f for f in ranked if binners[f].iv_ >= self.min_iv]

        # 2. greedy correlation pruning on WoE-transformed features (keep higher IV)
        woe = pd.DataFrame({f: binners[f].transform(df[f]) for f in candidates})
        kept: list[str] = []
        for f in candidates:
            if all(abs(woe[f].corr(woe[g])) <= self.max_corr for g in kept):
                kept.append(f)

        # 3. logistic regression on WoE features, with backward elimination:
        # WoE encodes risk direction, so every coefficient must be negative
        # (higher WoE = safer); wrong signs flag multicollinearity, and
        # insignificant features don't belong in a documented scorecard
        while True:
            X = sm.add_constant(woe[kept])
            model = sm.Logit(y.to_numpy(), X).fit(disp=0)
            coefs, pvals = model.params[kept], model.pvalues[kept]
            wrong_sign = coefs[coefs > 0]
            if not wrong_sign.empty:
                kept.remove(wrong_sign.index[0])
                continue
            insignificant = pvals[pvals > 0.05]
            if not insignificant.empty:
                kept.remove(insignificant.idxmax())
                continue
            break

        self.model_ = model
        self.binners_ = {f: binners[f] for f in kept}
        self.features_ = kept
        return self

    def _woe_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.binners_[f].transform(df[f]) for f in self.features_})

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(self._woe_matrix(df), has_constant="add")
        return np.asarray(self.model_.predict(X))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points scale: base_score at base_odds (good:bad), +pdo halves the odds of default."""
        p = self.predict_pd(df)
        log_odds_good = np.log((1 - p) / p)
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        return offset + factor * log_odds_good

    def summary(self) -> pd.DataFrame:
        rows = [
            {"feature": f, "iv": self.binners_[f].iv_,
             "coef": self.model_.params[f], "p_value": self.model_.pvalues[f]}
            for f in self.features_
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False)


def main() -> None:
    from sklearn.metrics import roc_auc_score  # oracle for the hand-rolled AUC

    df = pd.read_parquet("data/processed/panel.parquet")

    # right-censoring: a 1y outcome window needs 1y of observable history
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
    df = df[df["period_end"] <= cutoff]

    # out-of-time split
    train = df[df["fyear"] <= 2019]
    test = df[df["fyear"] >= 2020]
    print(f"train: {len(train)} rows ({int(train.default_within_1y.sum())} defaults, "
          f"{train.fyear.min()}-{train.fyear.max()})")
    print(f"test:  {len(test)} rows ({int(test.default_within_1y.sum())} defaults, "
          f"{test.fyear.min()}-{test.fyear.max()})")

    card = Scorecard().fit(train, train["default_within_1y"])
    print("\nselected features:")
    print(card.summary().to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for name, part in [("train (in-sample)", train), ("test (out-of-time)", test)]:
        y, p = part["default_within_1y"].to_numpy(), card.predict_pd(part)
        auc = roc_auc(y, p)
        assert abs(auc - roc_auc_score(y, p)) < 1e-9, "hand-rolled AUC disagrees with sklearn"
        print(f"\n{name}: AUC {auc:.3f} | Gini {gini(y, p):.3f} | "
              f"KS {ks_stat(y, p):.3f} | AR {accuracy_ratio(y, p):.3f}")

    # benchmark: Altman Z' on the same test rows
    sub = test.dropna(subset=["altman_z_prime"])
    z_auc = roc_auc(sub["default_within_1y"].to_numpy(), -sub["altman_z_prime"].to_numpy())
    print(f"\nAltman Z' benchmark on test: AUC {z_auc:.3f}")


if __name__ == "__main__":
    main()
