# Raven + Peregrine — Meta-Labeling for Algorithmic Trading on XAUUSD

## What this is

An algorithmic trading system for XAUUSD (gold) made of two pieces:

- **Peregrine** (`mql5/Peregrine.mq5`): an MQL5 Expert Advisor that trades Bollinger Band breakouts confirmed by H4 trend, ADX, and volume.
- **Raven** (`notebooks/RavenPeregrineXAUUSD.ipynb`): a *meta-labeling* classifier in Python (scikit-learn) that learns to predict whether a signal Peregrine already generated will win or lose, exported to ONNX to run embedded inside the EA.

This repository is not the story of "I built a profitable bot" — it is the story of how to validate (or rule out) whether an ML filter genuinely adds value on top of a rule-based strategy, including the real bugs found and fixed along the way.

## Why meta-labeling instead of price prediction

Predicting price directly with ML is, in practice, trying to extract signal from a process dominated by noise — serious quantitative research rarely does this with sustained success. *Meta-labeling* (a concept from López de Prado's *Advances in Financial Machine Learning*) instead takes a signal a technical rule already generated and asks a much more tractable question: **given that this opportunity already passed a reasonable technical filter, is it worth taking?** That is why Raven does not predict price or return — it predicts the probability that a specific Peregrine trade will be a winner.

See [`docs/01-architecture-meta-labeling.md`](docs/01-architecture-meta-labeling.md) for the full architecture (production EA + "collector" EA that logs features and real outcomes + training notebook + ONNX export).

## The three real bugs found (and why they matter more than the result)

During validation I found and fixed three distinct data leakage bugs — each one inflated the system's apparent profitability without being visible at a glance:

1. **Decision threshold tuned on the same set used to report results** — invalidated any out-of-sample performance estimate.
2. **The target variable (real trade outcome) leaking into the training features** — the model learned to read the answer instead of finding a market pattern, producing an artificial F1 of 1.0 (100% "accuracy").
3. **Feature misalignment between the trained model and the exported ONNX graph** — the production model ended up operating with its features shifted by one position, causing the AI filter to filter out essentially nothing in real production.

Full detail, with the exact before/after code for each fix, is in [`docs/02-data-leakage-findings.md`](docs/02-data-leakage-findings.md).

## How the result was validated: walk-forward, not a single backtest

A single backtest or a single forward-testing period is not enough to trust a trading system — it can be a lucky streak as much as an unlucky one. I implemented a **rolling-window walk-forward** validation: train on a chunk of history, choose the threshold on the chunk immediately after it (never seen during training), and evaluate on the chunk after that — repeating the process while sliding forward in time. Full methodology in [`docs/03-walk-forward-validation.md`](docs/03-walk-forward-validation.md).

## Honest result

**Raven's filter does not consistently outperform trading every Peregrine signal unfiltered.** Across a 5-window walk-forward validation, Raven "won" in 2 of 5 — and almost all of its accumulated advantage came from a single period where it avoided a real losing stretch.

This does not mean the project failed. It means the right question — *does the ML filter add real value, or does it just trim trades without genuine discrimination?* — finally has a rigorously measured answer, instead of a nice-looking, unfounded backtest number. Full analysis and next steps in [`docs/04-results-and-lessons-learned.md`](docs/04-results-and-lessons-learned.md).

## Repository structure

```
├── mql5/                          # Expert Advisors (MQL5)
│   ├── Peregrine.mq5               # Production EA (with ONNX inference)
│   └── Peregrine_collector.mq5     # Data collection EA (no AI, logging only)
├── notebooks/
│   └── RavenPeregrineXAUUSD.ipynb  # Training, validation, and ONNX export
├── src/
│   └── walk_forward_validation.py  # Rolling-window walk-forward validation
└── docs/
    ├── 01-architecture-meta-labeling.md
    ├── 02-data-leakage-findings.md
    ├── 03-walk-forward-validation.md
    └── 04-results-and-lessons-learned.md
```

## Stack

Python (scikit-learn, pandas, numpy), ONNX / skl2onnx, MQL5 / MetaTrader 5.

## Project status

Active research. The base strategy (Peregrine, without AI) is profitable in both backtest and genuine forward-testing; the meta-labeling layer (Raven) is under evaluation — see `docs/04-results-and-lessons-learned.md` for the next experiment in progress (pooling correlated assets to grow the training sample).
