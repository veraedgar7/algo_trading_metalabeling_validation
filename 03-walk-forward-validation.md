# Walk-forward validation (rolling window)

## The problem it solves

A single forward-testing period (say, 3 months out of sample) can be a lucky streak as much as an unlucky one — with only a few dozen trades, the statistical variance is too high to trust a single sample. In addition, a single fixed train/validation/test split says nothing about whether the system holds up across different market regimes over time.

## The idea

Instead of a single fixed split, the whole process of training → choosing a threshold → evaluating is repeated many times, sliding a time window forward:

```
Fold 1: [-------- train --------][-- val --][-- test --]
Fold 2:          [-------- train --------][-- val --][-- test --]
Fold 3:                   [-------- train --------][-- val --][-- test --]
                  ──────────────────────▶ time
```

Two design decisions matter here:

- **Rolling window, not expanding**: the training block has a fixed size and slides forward (dropping the oldest data as it advances) instead of growing indefinitely. This prevents the model from training on market regimes too old to still be relevant.
- **No fold ever evaluates on data it used to train or to choose its own threshold** — each fold's test block is strictly after its own training and validation blocks in time.

## What to look at in the results

- **Stability of the chosen threshold across folds**: if it varies little (for example, always between 0.30 and 0.45), that is a sign of a real pattern. If it jumps erratically between folds (0.30 in one, 0.75 in the next), that is a sign the "optimal threshold" is noise specific to each window, not a real signal.
- **Proportion of folds with a negative result**: one or two out of fifteen is expected for any real system; half or more indicates the absence of a real edge.
- **Fold-by-fold comparison against taking no filter at all**: a positive filtered result does not prove the filter adds value — you have to compare, fold by fold, whether filtering genuinely beats taking every signal unfiltered. A filter can be profitable in absolute terms and still add nothing over the base strategy without filtering.

## Result obtained in this project

Across 5 walk-forward windows, Raven's filter beat "no filter" in 2 of 5 folds. The total accumulated gain did favor Raven (roughly 14% more in total dollars), but that advantage was concentrated almost entirely in a single fold where the filter avoided a real losing stretch — not in a consistent, trade-by-trade discrimination of higher quality in the general case. See `04-results-and-lessons-learned.md` for the full interpretation.
