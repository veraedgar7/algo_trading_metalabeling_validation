"""
Rolling-window walk-forward validation for the Raven meta-labeling classifier.

Trains on a fixed-size, sliding window of past trades, chooses a decision
threshold using only the window immediately after it (never seen in
training), and evaluates honestly on the window after that -- strictly in
the future relative to that fold's training data. Repeats while sliding
forward in time.

`walk_forward_comparative` additionally computes, for every fold, the result
of taking every signal with no AI filter at all, to directly test whether
the filter adds value over the base rule-based strategy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def evaluate_threshold(df_slice, probs, threshold, profit_col="profit_real", min_trades=10):
    mask = probs >= threshold
    n_trades = mask.sum()
    if n_trades < min_trades:
        return n_trades, None
    result = df_slice.loc[mask, profit_col].sum()
    return n_trades, result


def walk_forward(
    df,
    feature_cols,
    target_col="target",
    profit_col="profit_real",
    train_window=400,
    val_window=80,
    test_window=60,
    step=60,
    model_params=None,
    threshold_grid=np.arange(0.30, 0.85, 0.05),
):
    n = len(df)
    start = 0
    fold_results = []
    fold_num = 0

    while start + train_window + val_window + test_window <= n:
        fold_num += 1
        train_slice = df.iloc[start : start + train_window]
        val_slice = df.iloc[start + train_window : start + train_window + val_window]
        test_slice = df.iloc[
            start + train_window + val_window : start + train_window + val_window + test_window
        ]

        X_tr, y_tr = train_slice[feature_cols], train_slice[target_col]
        X_va = val_slice[feature_cols]
        X_te = test_slice[feature_cols]

        model = RandomForestClassifier(random_state=42, class_weight="balanced", **(model_params or {}))
        model.fit(X_tr, y_tr)

        # choose the threshold using ONLY this fold's validation window
        prob_val = model.predict_proba(X_va)[:, 1]
        best_threshold, best_result = None, -np.inf
        for threshold in threshold_grid:
            n_v, result_v = evaluate_threshold(val_slice, prob_val, threshold, profit_col)
            if result_v is not None and result_v > best_result:
                best_result, best_threshold = result_v, threshold

        if best_threshold is None:
            best_threshold = 0.5  # fallback if no threshold gathers enough trades

        # honest evaluation on THIS fold's test window
        prob_test = model.predict_proba(X_te)[:, 1]
        n_test, test_result = evaluate_threshold(test_slice, prob_test, best_threshold, profit_col)

        fold_results.append(
            {
                "fold": fold_num,
                "train_start": start,
                "threshold_chosen": best_threshold,
                "test_trades": n_test,
                "test_result": test_result if test_result is not None else 0.0,
            }
        )

        start += step

    return pd.DataFrame(fold_results)


def walk_forward_comparative(
    df,
    feature_cols,
    target_col="target",
    profit_col="profit_real",
    train_window=400,
    val_window=80,
    test_window=60,
    step=30,
    model_params=None,
    threshold_grid=np.arange(0.30, 0.85, 0.05),
):
    n = len(df)
    start = 0
    fold_results = []
    fold_num = 0

    while start + train_window + val_window + test_window <= n:
        fold_num += 1
        train_slice = df.iloc[start : start + train_window]
        val_slice = df.iloc[start + train_window : start + train_window + val_window]
        test_slice = df.iloc[
            start + train_window + val_window : start + train_window + val_window + test_window
        ]

        X_tr, y_tr = train_slice[feature_cols], train_slice[target_col]
        X_va = val_slice[feature_cols]
        X_te = test_slice[feature_cols]

        model = RandomForestClassifier(random_state=42, class_weight="balanced", **(model_params or {}))
        model.fit(X_tr, y_tr)

        prob_val = model.predict_proba(X_va)[:, 1]
        best_threshold, best_result = None, -np.inf
        for threshold in threshold_grid:
            mask = prob_val >= threshold
            if mask.sum() >= 10:
                result_v = val_slice.loc[mask, profit_col].sum()
                if result_v > best_result:
                    best_result, best_threshold = result_v, threshold
        if best_threshold is None:
            best_threshold = 0.5

        # WITH the Raven filter
        prob_test = model.predict_proba(X_te)[:, 1]
        mask_test = prob_test >= best_threshold
        n_test_filtered = mask_test.sum()
        filtered_result = test_slice.loc[mask_test, profit_col].sum() if n_test_filtered > 0 else 0.0

        # WITHOUT any filter (every signal in the fold)
        n_test_unfiltered = len(test_slice)
        unfiltered_result = test_slice[profit_col].sum()

        fold_results.append(
            {
                "fold": fold_num,
                "train_start": start,
                "threshold_chosen": best_threshold,
                "trades_with_raven": n_test_filtered,
                "result_with_raven": filtered_result,
                "trades_no_filter": n_test_unfiltered,
                "result_no_filter": unfiltered_result,
                "raven_won": filtered_result > unfiltered_result,
            }
        )

        start += step

    return pd.DataFrame(fold_results)
