"""ORE input-file generation for the commodity CVA/PFE demo.

Generates the minimal self-contained config set ORE needs for a
single-currency, single-counterparty commodity netting set:

  - synthetic flat USD discount curve and gas forward curve (market.txt)
  - counterparty default curve as a flat hazard rate implied from the
    CreditLab model PD:  lambda = -ln(1 - PD_1y)
  - gas forward + fixed-price gas swap portfolio (one netting set)
  - cross-asset model: LGM 1F rates x Schwartz commodity, Sobol paths

All market data is synthetic and pedagogical. The XML layout follows the
ORE Examples (Exposure/MinimalSetup) for version 1.8.16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(y, m, min(d.day, days[m - 1]))


HAZARD_TENORS = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y")


@dataclass
class XvaInputs:
    """Parameters for one ORE CVA/PFE run (synthetic gas netting set)."""

    counterparty: str = "CPTY"
    pd_1y: float = 0.02              # CreditLab model 1y PD → flat hazard rate
    recovery: float = 0.4
    asof: date = date(2026, 7, 15)
    tenor_years: float = 3.0
    quantity_per_quarter: float = 250_000.0   # MMBtu per quarterly period
    spot: float = 3.50                        # USD/MMBtu
    contango: float = 0.02                    # forward curve drift per year
    rate: float = 0.04                        # flat USD zero rate
    sigma: float = 0.35                       # Schwartz commodity vol
    kappa: float = 0.3                        # Schwartz mean reversion
    samples: int = 2000
    quantile: float = 0.95
    seed: int = 42

    @property
    def hazard_rate(self) -> float:
        return -math.log(1.0 - self.pd_1y)

    @property
    def end_date(self) -> date:
        return _add_months(self.asof, int(self.tenor_years * 12))

    @property
    def swap_start(self) -> date:
        return _add_months(self.asof, 3)

    @property
    def grid(self) -> str:
        return f"{int(self.tenor_years * 4) + 4},3M"

    @property
    def forward_dates(self) -> list[date]:
        return [
            _add_months(self.asof, 3 * i)
            for i in range(1, int(self.tenor_years * 4) + 9)
        ]

    @property
    def fixed_price(self) -> float:
        """Swap fixed price ~ATM against the mid-life forward."""
        return self.spot * math.exp(self.contango * self.tenor_years / 2)

    @property
    def forward_strike(self) -> float:
        """Forward strike at the maturity-date forward (zero initial MtM)."""
        return self.spot * math.exp(self.contango * self.tenor_years)


COM_CURVE = "GAS_USD"
COM_NAME = "COMDTY_GAS_USD"
COM_QUOTE = "GAS"


def market_txt(p: XvaInputs) -> str:
    d = p.asof.strftime("%Y%m%d")
    lines = [
        f"{d} ZERO/RATE/USD/USD-FLAT/A365/1Y {p.rate}",
        f"{d} SWAPTION/RATE_NVOL/USD/1Y/1Y/ATM 0.0050",
        f"{d} COMMODITY/PRICE/{COM_QUOTE}/USD {p.spot}",
    ]
    for fd in p.forward_dates:
        t = (fd - p.asof).days / 365.0
        px = p.spot * math.exp(p.contango * t)
        lines.append(f"{d} COMMODITY_FWD/PRICE/{COM_QUOTE}/USD/{fd.isoformat()} {px:.4f}")
    for t in ("1Y", "5Y"):
        lines.append(f"{d} COMMODITY_OPTION/RATE_LNVOL/{COM_QUOTE}/USD/{t}/ATM/AtmFwd {p.sigma}")
    lines.append(f"{d} RECOVERY_RATE/RATE/{p.counterparty}/SR/USD {p.recovery}")
    for t in HAZARD_TENORS:
        lines.append(f"{d} HAZARD_RATE/RATE/{p.counterparty}/SR/USD/{t} {p.hazard_rate:.6f}")
    return "\n".join(lines) + "\n"


def conventions_xml(p: XvaInputs) -> str:
    return """<?xml version="1.0"?>
<Conventions>
  <Zero>
    <Id>USD-ZERO-CONVENTIONS-TENOR-BASED</Id>
    <TenorBased>true</TenorBased>
    <DayCounter>A365</DayCounter>
    <Compounding>Continuous</Compounding>
    <CompoundingFrequency>Daily</CompoundingFrequency>
    <TenorCalendar>US</TenorCalendar>
    <SpotLag>0</SpotLag>
    <SpotCalendar>US</SpotCalendar>
    <RollConvention>Following</RollConvention>
    <EOM>false</EOM>
  </Zero>
  <CDS>
    <Id>CDS-STANDARD-CONVENTIONS</Id>
    <SettlementDays>0</SettlementDays>
    <Calendar>WeekendsOnly</Calendar>
    <Frequency>Quarterly</Frequency>
    <PaymentConvention>ModifiedFollowing</PaymentConvention>
    <Rule>TwentiethIMM</Rule>
    <DayCounter>A360</DayCounter>
    <SettlesAccrual>true</SettlesAccrual>
    <PaysAtDefaultTime>true</PaysAtDefaultTime>
  </CDS>
  <SwapIndex>
    <Id>USD-CMS-1Y</Id>
    <Conventions>USD-3M-SWAP-CONVENTIONS</Conventions>
  </SwapIndex>
  <SwapIndex>
    <Id>USD-CMS-30Y</Id>
    <Conventions>USD-3M-SWAP-CONVENTIONS</Conventions>
  </SwapIndex>
  <Swap>
    <Id>USD-3M-SWAP-CONVENTIONS</Id>
    <FixedCalendar>US</FixedCalendar>
    <FixedFrequency>Semiannual</FixedFrequency>
    <FixedConvention>MF</FixedConvention>
    <FixedDayCounter>30/360</FixedDayCounter>
    <Index>USD-LIBOR-3M</Index>
  </Swap>
</Conventions>
"""


def curveconfig_xml(p: XvaInputs) -> str:
    fwd_quotes = "\n".join(
        f"        <Quote>COMMODITY_FWD/PRICE/{COM_QUOTE}/USD/{fd.isoformat()}</Quote>"
        for fd in p.forward_dates
    )
    hz_quotes = "\n".join(
        f"        <Quote>HAZARD_RATE/RATE/{p.counterparty}/SR/USD/{t}</Quote>"
        for t in HAZARD_TENORS
    )
    return f"""<?xml version="1.0"?>
<CurveConfiguration>
  <YieldCurves>
    <YieldCurve>
      <CurveId>USD-FLAT</CurveId>
      <CurveDescription>Flat USD discount curve</CurveDescription>
      <Currency>USD</Currency>
      <DiscountCurve/>
      <Segments>
        <Direct>
          <Type>Zero</Type>
          <Quotes>
            <Quote>ZERO/RATE/USD/USD-FLAT/A365/1Y</Quote>
          </Quotes>
          <Conventions>USD-ZERO-CONVENTIONS-TENOR-BASED</Conventions>
        </Direct>
      </Segments>
    </YieldCurve>
  </YieldCurves>
  <SwaptionVolatilities>
    <SwaptionVolatility>
      <CurveId>USD_SWPTN</CurveId>
      <CurveDescription>Flat USD normal swaption vol</CurveDescription>
      <Dimension>ATM</Dimension>
      <VolatilityType>Normal</VolatilityType>
      <Extrapolation>Flat</Extrapolation>
      <DayCounter>Actual/365 (Fixed)</DayCounter>
      <Calendar>US</Calendar>
      <BusinessDayConvention>Following</BusinessDayConvention>
      <OptionTenors>1Y</OptionTenors>
      <SwapTenors>1Y</SwapTenors>
      <ShortSwapIndexBase>USD-CMS-1Y</ShortSwapIndexBase>
      <SwapIndexBase>USD-CMS-30Y</SwapIndexBase>
    </SwaptionVolatility>
  </SwaptionVolatilities>
  <DefaultCurves>
    <DefaultCurve>
      <CurveId>{p.counterparty}_SR_USD</CurveId>
      <CurveDescription>Flat hazard rate implied from CreditLab 1y PD</CurveDescription>
      <Currency>USD</Currency>
      <Type>HazardRate</Type>
      <DiscountCurve/>
      <DayCounter>A365</DayCounter>
      <RecoveryRate>RECOVERY_RATE/RATE/{p.counterparty}/SR/USD</RecoveryRate>
      <Quotes>
{hz_quotes}
      </Quotes>
      <Conventions>CDS-STANDARD-CONVENTIONS</Conventions>
    </DefaultCurve>
  </DefaultCurves>
  <CommodityCurves>
    <CommodityCurve>
      <CurveId>{COM_CURVE}</CurveId>
      <CurveDescription>Synthetic gas forward curve</CurveDescription>
      <Currency>USD</Currency>
      <SpotQuote>COMMODITY/PRICE/{COM_QUOTE}/USD</SpotQuote>
      <Quotes>
{fwd_quotes}
      </Quotes>
      <DayCounter>A365</DayCounter>
      <InterpolationMethod>Linear</InterpolationMethod>
      <Extrapolation>true</Extrapolation>
    </CommodityCurve>
  </CommodityCurves>
  <CommodityVolatilities>
    <CommodityVolatility>
      <CurveId>{COM_CURVE}_VOLS</CurveId>
      <CurveDescription>Flat lognormal gas vol</CurveDescription>
      <Currency>USD</Currency>
      <Curve>
        <Quotes>
          <Quote>COMMODITY_OPTION/RATE_LNVOL/{COM_QUOTE}/USD/1Y/ATM/AtmFwd</Quote>
          <Quote>COMMODITY_OPTION/RATE_LNVOL/{COM_QUOTE}/USD/5Y/ATM/AtmFwd</Quote>
        </Quotes>
        <Interpolation>Linear</Interpolation>
        <Extrapolation>Flat</Extrapolation>
      </Curve>
      <DayCounter>A365</DayCounter>
      <Calendar>USD</Calendar>
    </CommodityVolatility>
  </CommodityVolatilities>
</CurveConfiguration>
"""


def todaysmarket_xml(p: XvaInputs) -> str:
    return f"""<?xml version="1.0"?>
<TodaysMarket>
  <Configuration id="default">
    <DiscountingCurvesId>default</DiscountingCurvesId>
    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
    <SwapIndexCurvesId>default</SwapIndexCurvesId>
    <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
    <CommodityCurvesId>default</CommodityCurvesId>
    <CommodityVolatilitiesId>default</CommodityVolatilitiesId>
    <DefaultCurvesId>default</DefaultCurvesId>
  </Configuration>
  <DiscountingCurves id="default">
    <DiscountingCurve currency="USD">Yield/USD/USD-FLAT</DiscountingCurve>
  </DiscountingCurves>
  <IndexForwardingCurves id="default">
    <Index name="USD-LIBOR-3M">Yield/USD/USD-FLAT</Index>
    <Index name="USD-SOFR">Yield/USD/USD-FLAT</Index>
  </IndexForwardingCurves>
  <SwapIndexCurves id="default">
    <SwapIndex name="USD-CMS-1Y">
      <Discounting>USD-LIBOR-3M</Discounting>
    </SwapIndex>
    <SwapIndex name="USD-CMS-30Y">
      <Discounting>USD-LIBOR-3M</Discounting>
    </SwapIndex>
  </SwapIndexCurves>
  <SwaptionVolatilities id="default">
    <SwaptionVolatility currency="USD">SwaptionVolatility/USD/USD_SWPTN</SwaptionVolatility>
  </SwaptionVolatilities>
  <CommodityCurves id="default">
    <CommodityCurve name="{COM_NAME}">Commodity/USD/{COM_CURVE}</CommodityCurve>
  </CommodityCurves>
  <CommodityVolatilities id="default">
    <CommodityVolatility name="{COM_NAME}">CommodityVolatility/USD/{COM_CURVE}_VOLS</CommodityVolatility>
  </CommodityVolatilities>
  <DefaultCurves id="default">
    <DefaultCurve name="{p.counterparty}">Default/USD/{p.counterparty}_SR_USD</DefaultCurve>
  </DefaultCurves>
</TodaysMarket>
"""


def pricingengine_xml(p: XvaInputs) -> str:
    return """<?xml version="1.0"?>
<PricingEngines>
  <Product type="CommodityForward">
    <Model>DiscountedCashflows</Model>
    <ModelParameters/>
    <Engine>DiscountingCommodityForwardEngine</Engine>
    <EngineParameters/>
  </Product>
  <Product type="CommoditySwap">
    <Model>DiscountedCashflows</Model>
    <ModelParameters/>
    <Engine>CommoditySwapEngine</Engine>
    <EngineParameters/>
  </Product>
  <Product type="Swap">
    <Model>DiscountedCashflows</Model>
    <ModelParameters/>
    <Engine>DiscountingSwapEngine</Engine>
    <EngineParameters/>
  </Product>
</PricingEngines>
"""


def portfolio_xml(p: XvaInputs) -> str:
    return f"""<?xml version="1.0"?>
<Portfolio>
  <Trade id="GasForward">
    <TradeType>CommodityForward</TradeType>
    <Envelope>
      <CounterParty>{p.counterparty}</CounterParty>
      <NettingSetId>{p.counterparty}</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <CommodityForwardData>
      <Position>Long</Position>
      <Maturity>{p.end_date.isoformat()}</Maturity>
      <Name>{COM_NAME}</Name>
      <Currency>USD</Currency>
      <Strike>{p.forward_strike:.4f}</Strike>
      <Quantity>{p.quantity_per_quarter * 4:.0f}</Quantity>
    </CommodityForwardData>
  </Trade>
  <Trade id="GasFixedPriceSwap">
    <TradeType>CommoditySwap</TradeType>
    <Envelope>
      <CounterParty>{p.counterparty}</CounterParty>
      <NettingSetId>{p.counterparty}</NettingSetId>
      <AdditionalFields/>
    </Envelope>
    <SwapData>
      <LegData>
        <LegType>CommodityFixed</LegType>
        <Payer>true</Payer>
        <Currency>USD</Currency>
        <PaymentLag>2</PaymentLag>
        <PaymentConvention>Following</PaymentConvention>
        <PaymentCalendar>US</PaymentCalendar>
        <CommodityFixedLegData>
          <Quantities><Quantity>{p.quantity_per_quarter:.0f}</Quantity></Quantities>
          <Prices><Price>{p.fixed_price:.4f}</Price></Prices>
        </CommodityFixedLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{p.swap_start.isoformat()}</StartDate>
            <EndDate>{p.end_date.isoformat()}</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>NullCalendar</Calendar>
            <Convention>Unadjusted</Convention>
            <TermConvention>Unadjusted</TermConvention>
            <Rule>Backward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
      <LegData>
        <LegType>CommodityFloating</LegType>
        <Payer>false</Payer>
        <Currency>USD</Currency>
        <PaymentLag>2</PaymentLag>
        <PaymentConvention>Following</PaymentConvention>
        <PaymentCalendar>US</PaymentCalendar>
        <CommodityFloatingLegData>
          <Name>{COM_NAME}</Name>
          <PriceType>Spot</PriceType>
          <Quantities><Quantity>{p.quantity_per_quarter:.0f}</Quantity></Quantities>
          <IsAveraged>false</IsAveraged>
        </CommodityFloatingLegData>
        <ScheduleData>
          <Rules>
            <StartDate>{p.swap_start.isoformat()}</StartDate>
            <EndDate>{p.end_date.isoformat()}</EndDate>
            <Tenor>3M</Tenor>
            <Calendar>NullCalendar</Calendar>
            <Convention>Unadjusted</Convention>
            <TermConvention>Unadjusted</TermConvention>
            <Rule>Backward</Rule>
          </Rules>
        </ScheduleData>
      </LegData>
    </SwapData>
  </Trade>
</Portfolio>
"""


def netting_xml(p: XvaInputs) -> str:
    return f"""<?xml version="1.0"?>
<NettingSetDefinitions>
  <NettingSet>
    <NettingSetId>{p.counterparty}</NettingSetId>
    <ActiveCSAFlag>false</ActiveCSAFlag>
    <CSADetails/>
  </NettingSet>
</NettingSetDefinitions>
"""


def simulation_xml(p: XvaInputs) -> str:
    return f"""<?xml version="1.0"?>
<Simulation>
  <Parameters>
    <Discretization>Exact</Discretization>
    <Grid>{p.grid}</Grid>
    <Calendar>US</Calendar>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{p.seed}</Seed>
    <Samples>{p.samples}</Samples>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>USD</DomesticCcy>
    <Currencies>
      <Currency>USD</Currency>
    </Currencies>
    <Commodities>
      <Commodity>{COM_NAME}</Commodity>
    </Commodities>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <InterestRateModels>
      <LGM ccy="USD">
        <CalibrationType>Bootstrap</CalibrationType>
        <Volatility>
          <Calibrate>Y</Calibrate>
          <VolatilityType>Hagan</VolatilityType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.01</InitialValue>
        </Volatility>
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.03</InitialValue>
        </Reversion>
        <CalibrationSwaptions>
          <Expiries>1Y</Expiries>
          <Terms>1Y</Terms>
          <Strikes/>
        </CalibrationSwaptions>
        <ParameterTransformation>
          <ShiftHorizon>0.0</ShiftHorizon>
          <Scaling>1.0</Scaling>
        </ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <CommodityModels>
      <CommoditySchwartz name="{COM_NAME}">
        <Currency>USD</Currency>
        <CalibrationType>None</CalibrationType>
        <Sigma>
          <Calibrate>false</Calibrate>
          <InitialValue>{p.sigma}</InitialValue>
        </Sigma>
        <Kappa>
          <Calibrate>false</Calibrate>
          <InitialValue>{p.kappa}</InitialValue>
        </Kappa>
        <Seasonality>
          <Calibrate>false</Calibrate>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>0.0</InitialValue>
        </Seasonality>
        <CalibrationOptions>
          <Expiries/>
          <Strikes/>
        </CalibrationOptions>
        <DriftFreeState>false</DriftFreeState>
      </CommoditySchwartz>
    </CommodityModels>
    <InstantaneousCorrelations>
      <Correlation factor1="IR:USD" factor2="COM:{COM_NAME}">0.0</Correlation>
    </InstantaneousCorrelations>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>USD</BaseCurrency>
    <Currencies>
      <Currency>USD</Currency>
    </Currencies>
    <YieldCurves>
      <Configuration>
        <Tenors>3M,6M,1Y,2Y,3Y,5Y,7Y,10Y</Tenors>
        <Interpolation>LogLinear</Interpolation>
        <Extrapolation>Y</Extrapolation>
      </Configuration>
    </YieldCurves>
    <Indices>
      <Index>USD-LIBOR-3M</Index>
      <Index>USD-SOFR</Index>
    </Indices>
    <SwapIndices>
      <SwapIndex>
        <Name>USD-CMS-1Y</Name>
        <DiscountingIndex>USD-LIBOR-3M</DiscountingIndex>
      </SwapIndex>
      <SwapIndex>
        <Name>USD-CMS-30Y</Name>
        <DiscountingIndex>USD-LIBOR-3M</DiscountingIndex>
      </SwapIndex>
    </SwapIndices>
    <DefaultCurves>
      <Names/>
      <Tenors>6M,1Y,2Y</Tenors>
    </DefaultCurves>
    <SwaptionVolatilities>
      <ReactionToTimeDecay>ForwardVariance</ReactionToTimeDecay>
      <Currencies>
        <Currency>USD</Currency>
      </Currencies>
      <Expiries>1Y</Expiries>
      <Terms>1Y</Terms>
    </SwaptionVolatilities>
    <Commodities>
      <Simulate>true</Simulate>
      <Names>
        <Name>{COM_NAME}</Name>
      </Names>
      <Tenors>3M,6M,1Y,2Y,3Y,5Y</Tenors>
      <DayCounters>
        <DayCounter name="">A365</DayCounter>
      </DayCounters>
    </Commodities>
    <CommodityVolatilities>
      <Simulate>false</Simulate>
      <ReactionToTimeDecay>ConstantVariance</ReactionToTimeDecay>
      <Names>
        <Name id="{COM_NAME}">
          <Expiries>1Y,5Y</Expiries>
        </Name>
      </Names>
      <DayCounter>A365</DayCounter>
    </CommodityVolatilities>
    <AggregationScenarioDataCurrencies>
      <Currency>USD</Currency>
    </AggregationScenarioDataCurrencies>
    <AggregationScenarioDataIndices>
      <Index>USD-LIBOR-3M</Index>
    </AggregationScenarioDataIndices>
  </Market>
</Simulation>
"""


def ore_xml(p: XvaInputs, work_dir: str) -> str:
    return f"""<?xml version="1.0"?>
<ORE>
  <Setup>
    <Parameter name="asofDate">{p.asof.isoformat()}</Parameter>
    <Parameter name="inputPath">{work_dir}</Parameter>
    <Parameter name="outputPath">{work_dir}/Output</Parameter>
    <Parameter name="logFile">log.txt</Parameter>
    <Parameter name="logMask">31</Parameter>
    <Parameter name="marketDataFile">market.txt</Parameter>
    <Parameter name="fixingDataFile">fixings.txt</Parameter>
    <Parameter name="implyTodaysFixings">N</Parameter>
    <Parameter name="curveConfigFile">curveconfig.xml</Parameter>
    <Parameter name="conventionsFile">conventions.xml</Parameter>
    <Parameter name="marketConfigFile">todaysmarket.xml</Parameter>
    <Parameter name="pricingEnginesFile">pricingengine.xml</Parameter>
    <Parameter name="portfolioFile">portfolio.xml</Parameter>
    <Parameter name="observationModel">None</Parameter>
  </Setup>
  <Markets>
    <Parameter name="lgmcalibration">default</Parameter>
    <Parameter name="fxcalibration">default</Parameter>
    <Parameter name="pricing">default</Parameter>
    <Parameter name="simulation">default</Parameter>
  </Markets>
  <Analytics>
    <Analytic type="npv">
      <Parameter name="active">Y</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
      <Parameter name="outputFileName">npv.csv</Parameter>
    </Analytic>
    <Analytic type="simulation">
      <Parameter name="active">Y</Parameter>
      <Parameter name="simulationConfigFile">simulation.xml</Parameter>
      <Parameter name="pricingEnginesFile">pricingengine.xml</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="aggregationScenarioDataFileName">scenariodata.csv.gz</Parameter>
    </Analytic>
    <Analytic type="xva">
      <Parameter name="active">Y</Parameter>
      <Parameter name="csaFile">netting.xml</Parameter>
      <Parameter name="cubeFile">cube.csv.gz</Parameter>
      <Parameter name="scenarioFile">scenariodata.csv.gz</Parameter>
      <Parameter name="baseCurrency">USD</Parameter>
      <Parameter name="exposureProfiles">Y</Parameter>
      <Parameter name="exposureProfilesByTrade">Y</Parameter>
      <Parameter name="quantile">{p.quantile}</Parameter>
      <Parameter name="calculationType">Symmetric</Parameter>
      <Parameter name="allocationMethod">None</Parameter>
      <Parameter name="marginalAllocationLimit">1.0</Parameter>
      <Parameter name="exerciseNextBreak">N</Parameter>
      <Parameter name="cva">Y</Parameter>
      <Parameter name="dva">N</Parameter>
      <Parameter name="fva">N</Parameter>
      <Parameter name="colva">N</Parameter>
      <Parameter name="collateralFloor">N</Parameter>
    </Analytic>
  </Analytics>
</ORE>
"""


def write_all(p: XvaInputs, work_dir: str) -> None:
    """Write the full ORE input set into ``work_dir``."""
    import os

    os.makedirs(work_dir, exist_ok=True)
    files = {
        "market.txt": market_txt(p),
        "fixings.txt": "",
        "conventions.xml": conventions_xml(p),
        "curveconfig.xml": curveconfig_xml(p),
        "todaysmarket.xml": todaysmarket_xml(p),
        "pricingengine.xml": pricingengine_xml(p),
        "portfolio.xml": portfolio_xml(p),
        "netting.xml": netting_xml(p),
        "simulation.xml": simulation_xml(p),
        "ore.xml": ore_xml(p, work_dir),
    }
    for name, content in files.items():
        with open(os.path.join(work_dir, name), "w") as f:
            f.write(content)
