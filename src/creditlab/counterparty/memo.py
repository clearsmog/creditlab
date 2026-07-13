"""Front Office–facing credit memo text for counterparty limit recommendations."""

from __future__ import annotations

from creditlab.counterparty.limits import LimitRecommendation


def format_credit_memo(
    rec: LimitRecommendation,
    *,
    product: str = "physical & financial gas / power / LNG hedges",
    current_exposure_usd: float = 0.0,
    proposed_deal_pfe_usd: float = 0.0,
) -> str:
    """Plain-language credit opinion a FO desk can action."""
    lim = rec.recommended_limit_usd
    post = current_exposure_usd + proposed_deal_pfe_usd
    util = post / lim if lim > 0 else float("inf")
    decision = (
        "APPROVE unsecured capacity as proposed"
        if lim > 0 and util <= 1.0
        else (
            "CONDITIONAL — structure with credit support / shorter tenor"
            if lim > 0
            else "DECLINE unsecured — collateral / prepay / LC only"
        )
    )

    flags = rec.ratio_flags
    ratio_lines = (
        f"- Leverage: **{flags.leverage}** · Interest coverage: **{flags.interest_coverage}**\n"
        f"- Current ratio: **{flags.current_ratio}** · ROA: **{flags.roa}**"
    )
    if flags.notes:
        ratio_lines += "\n- Notes: " + "; ".join(flags.notes)

    docs = "\n".join(f"- {d}" for d in rec.documentation)
    rationale = "\n".join(f"- {r}" for r in rec.rationale)

    return f"""# Counterparty credit memo (desk draft)

**Counterparty:** {rec.name} ({rec.ticker})  
**Product context:** {product}  
**Internal rating:** {rec.rating} · **Model 1y PD:** {rec.pd_1y:.2%}  
**KYC status (demo):** {rec.kyc_status.upper()}

## Recommendation

**{decision}**

| Item | Amount |
| --- | ---: |
| Proposed unsecured limit | ${lim:,.0f} |
| Max tenor | {rec.max_tenor_years:.1f} years |
| Current exposure (input) | ${current_exposure_usd:,.0f} |
| Proposed deal PFE add-on (input) | ${proposed_deal_pfe_usd:,.0f} |
| Post-deal utilisation | {util:.0%} |

## Financial statement flags

{ratio_lines}

## Rationale

{rationale}

## Documentation & credit support

{docs}

## FO actions

1. Confirm KYC / CDD is current under house process before onboarding or renewal.
2. Route limit to credit committee if above desk authority or KYC = escalate.
3. Book only within limit; monitor MtM + PFE; escalate breaches immediately.
4. Prefer risk-reducing structures (netting, CSA thresholds, shorter tenor) if commercial pressure exceeds unsecured capacity.

---
*Illustrative CreditLab output — not a live SEFE / house credit decision.*
"""
