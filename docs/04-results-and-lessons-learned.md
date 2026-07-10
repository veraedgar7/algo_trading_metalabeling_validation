# Results and lessons learned

## The result, unembellished

After fixing the three data leakage bugs documented in `02-data-leakage-findings.md` and validating with walk-forward testing (`03-walk-forward-validation.md`), the central finding is:

**The meta-labeling filter (Raven) does not consistently outperform trading every Peregrine signal unfiltered.**

Across 5 walk-forward validation windows:

| Fold | With Raven filter | No filter | Did Raven win? |
|---|---|---|---|
| 1 | $33,599.40 | $35,100.37 | No |
| 2 | $13,690.84 | $9,467.31 | Yes |
| 3 | $16,138.60 | $17,183.27 | No |
| 4 | $16,678.67 | $16,678.67 | Tie (filtered nothing) |
| 5 | $6,693.15 | -$2,110.20 | Yes |

The total accumulated gain favors Raven (roughly 14% more in total dollars), but that advantage comes almost entirely from fold 5, where the filter prevented a period with a net loss from happening. In the other folds, the filter barely acts at all (it lets through 90-100% of the signals) or slightly hurts the result.

## The honest interpretation

This is not evidence that the project failed — it is evidence that **the AI filter, as currently built, works more like an occasional emergency brake for bad regimes than a consistent discriminator of signal quality.** That is a different — and more modest — role than the one originally intended.

The root cause identified is not the model itself (DecisionTree, RandomForest, XGBoost, and LightGBM were all tried, with hyperparameter search via `TimeSeriesSplit`) — it is that **the base signal generator (Peregrine) produces very few labeled events per unit of time**, and that volume of data is not enough for a classifier to learn a fine, stable discrimination pattern beyond catching extreme cases.

## What was tried, and what is still untested

**Tried:**
- Comparison of 4 model families (trees, ensembles, gradient boosting).
- Two objective functions for choosing the threshold: total profit sum vs. average profit per trade — neither solves the root problem, each just moves the trade-off between trade volume and selectivity.
- Rolling-window walk-forward validation, always compared against the "no filter" alternative.

**Next experiment (not yet tried)**: instead of continuing to tune the model on the same limited dataset, **expand the training set by pooling events from correlated assets** (silver, platinum — same precious metals family, same session/liquidity profile as gold) to increase the number of contemporaneous events without resorting to history from too old a market regime. This is the hypothesis most likely to address the identified root cause, rather than continuing to iterate on hyperparameters with the same scarce data.

## Why this result, documented this way, has value

A positive result without this level of scrutiny would be less trustworthy than this negative result with real scrutiny — most "ML applied to trading" projects circulating publicly go through none of these three validations. Honestly finding and documenting that an approach does not work as expected, with reproducible evidence of why, is exactly the kind of rigor that distinguishes a serious applied ML project from a nice-looking, unfounded backtest.
