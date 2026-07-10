# Architecture: meta-labeling applied to a trading EA

## The general pattern

```
┌─────────────────┐     ┌───────────────────────┐         ┌────────────────────┐
│  Peregrine      │     │ Peregrine_collector   │         │   Raven (Python)   │
│ (breakout rules:│ ─▶ │ (same EA, but logs    │ ───────▶│  trains a binary   │
│  BB breakout +  │     │  features + the real  │         │  classifier on     │
│  H4 trend+ADX+  │     │  outcome of every     │         │  those events      │
│  volume)        │     │  trade)               │         └──────────┬─────────┘
└─────────────────┘     └───────────────────────┘                    │
                                                                     ▼
                                                             exports to ONNX
                                                                     │
                                                                     ▼
                                                        ┌──────────────────────┐
                                                        │  Peregrine.mq5       │
                                                        │  (production): runs  │
                                                        │  the rule + queries the │
                                                        │  ONNX model before   │
                                                        │  executing the trade │
                                                        └──────────────────────┘
```

## Why this architecture, and not "predict the price"

Predicting the next candle's price (or return) tries to extract signal from a process where noise dominates — the genuine predictive power that serious quantitative research finds on raw returns is marginal even at the best systematic funds. The meta-labeling pattern inverts the problem: instead of "where is price going?", it asks "given that a reasonable technical rule already identified an opportunity, is this specific instance one of the ones that usually works?". It is a problem with far more usable signal, because it is already bounded by domain knowledge (the base rule).

## The pieces

### 1. Peregrine — signal generator (base rule)
Bollinger Band breakout, confirmed by:
- H4 trend (higher timeframe)
- ADX above a threshold (confirms a trend is present, not a range)
- Relative volume filter
- Risk management with ATR-based SL/TP

### 2. Peregrine_collector — training data collection
The same signal generator as Peregrine, but without the AI filter — its only purpose is to **log, for every real trade, the features at entry time and the real outcome at close** (won/lost, and the real profit in dollars). This log is Raven's training dataset.

Logged features: `bb_norm` (position within the band), `adx`, `rsi`, `vol_rel` (relative volume), `hour_sin`/`hour_cos` (time of day, cyclically encoded), `dist_MA` (distance to the moving average), `stch` (candle stretch).

### 3. Raven — the meta-labeling classifier
A Python notebook that:
- Aggregates the log by trade (`ticket`), taking the entry-time features and the close-time outcome.
- Splits the data **chronologically** into train / validation / test — never randomly, since shuffling time order in market data invalidates any validation.
- Trains and compares several classifiers (DecisionTree, RandomForest, XGBoost, LightGBM) with `TimeSeriesSplit` for hyperparameter search.
- Chooses a decision probability threshold using **only** the validation set (see `02-data-leakage-findings.md` for why this is not optional).
- Exports the chosen model to ONNX for inference in MQL5.

### 4. Peregrine (production) — model consumer
The production EA runs the same rule-based logic as the collector, but before executing each trade it queries the embedded ONNX model: if the predicted win probability is below the threshold chosen during validation, the signal is discarded.
