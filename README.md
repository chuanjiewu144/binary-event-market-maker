# Binary Event Market Maker

A research implementation of a solvency-first market maker for binary
event contracts.

The Version 6 strategy combines:

- rolling stochastic-model calibration;
- analytic and deterministic Monte Carlo pricing;
- covariance-aware inventory management;
- competitive two-sided RFQ quoting;
- directional informed-flow protection for FOK orders;
- exact maximum-loss cash accounting; and
- state-level caching for controlled execution time.

The primary objective is to avoid bankruptcy under admissible execution
sequences. Subject to this constraint, the strategy seeks to maximise
spread capture, realised PnL and competitive rank.

## Repository structure

- `submission/market_maker_v6.py`: exact single-class challenge submission.
- `docs/report/main.tex`: Overleaf-compatible technical report source.
- `docs/report/report.pdf`: compiled ten-page technical report.
- `tests/`: planned automated validation suite.
- `.github/workflows/`: automated syntax and LaTeX compilation checks.

## Strategy summary

### RFQ orders

The strategy produces a two-sided bid and offer. Prices depend on fair
value, estimation uncertainty, live volatility, counterparty toxicity,
recent order flow and portfolio covariance.

Displayed quantities are controlled separately from prices using
maximum-loss cash capacity, inventory direction and concentration limits.

### FOK orders

The customer reveals side, price and total quantity. The strategy stresses
the estimated payoff probability in the direction suggested by unusual
price, size, short tenor, repeated flow and counterparty toxicity.

The complete order is accepted only if:

1. stressed edge is sufficient;
2. maximum loss is affordable;
3. hard position limits are satisfied; and
4. the FOK loss budget is satisfied.

## Technical report

The complete mathematical model and strategy description are available in:

- [LaTeX source](docs/report/main.tex)
- [Compiled PDF](docs/report/report.pdf)

## Validation status

The current repository includes syntax and LaTeX compilation checks.

Automated unit, solvency, regression and scenario tests have not yet been
implemented. The planned test coverage is documented under `tests/`.

## Limitations

This project was developed for a controlled market-making simulation.
It is not a production trading system, does not model real exchange
latency or market impact, and is not investment advice.