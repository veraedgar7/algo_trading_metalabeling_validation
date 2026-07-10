# The three data leakage bugs found (and how they were fixed)

Each of these inflated the system's apparent profitability without being visible in a superficial review of the notebook. Documenting them is more valuable than the final result — it is the part of the process that demonstrates genuine validation rigor.

## 1. Decision threshold tuned on the test set

**The problem**: the probability threshold that decides whether a signal is taken or not is itself a hyperparameter. The first version of the notebook chose it by scanning several candidate values and keeping the one that maximized the economic result — **evaluated directly against the same data that was later reported as the "out-of-sample result."** The moment any parameter is tuned by looking at a dataset's outcome, that dataset stops being a fair test — it becomes part of the training process, even if the model itself was never refit on it.

**The fix**: split the data into three chronological parts instead of two:

```python
n_rows = len(df)
train_end = int(n_rows * 0.60)
val_end   = int(n_rows * 0.80)

train_set = df.iloc[:train_end]         # fit the model
val_set   = df.iloc[train_end:val_end]  # choose the threshold
test_set  = df.iloc[val_end:]           # evaluate ONCE, never before
```

The threshold is searched for only against `val_set`. The number reported as "expected performance" comes from applying that threshold, already frozen, to `test_set` — which was never touched before that point.

## 2. The target variable leaking into the training features

**The problem**: when `profit_real` (the real dollar outcome of each trade) was added to make the threshold's economic simulation more honest, that same column accidentally stayed inside the model's input features (`X_train`). Since the target (`target = 1 if profit_real > 0`) is derived directly from that column, the model had access to the answer disguised as a question — any classifier finds that pattern trivially and reports F1 = 1.0 (100% accuracy), without having learned any real market pattern.

**The fix**: declare the feature columns explicitly, instead of excluding only the target column:

```python
# Before (let profit_real slip through unintentionally):
X_train = train_set.drop(columns=['target'])

# After — explicit feature list:
FEATURE_COLS = ['bb_norm', 'adx', 'rsi', 'vol_rel', 'hour_sin', 'hour_cos', 'dist_MA', 'stch']
X_train = train_set[FEATURE_COLS]
profit_train = train_set['profit_real']  # kept separate, never fed to the model
```

This leak appeared **twice** in the same project — once in the main training cell, and again, independently, in the cell that retrained the final "production model" before exporting it to ONNX. Both required the same fix applied separately, because each cell rebuilt `X`/`y` from scratch.

## 3. Feature misalignment between the trained model and the exported ONNX graph

**The problem**: as a direct consequence of leak #2 in the production cell, the final model was trained on 9 columns (8 real features + `profit_real`), but the ONNX graph was exported declaring an 8-feature input. The production model ended up reading every real feature shifted by one position relative to what MQL5 actually sent — the observable result was that the AI filter **let 100% of the signals through**, identical to having no filter at all, without raising any visible error (the real error only surfaced in the MetaTrader execution logs once the input-side leak was partially fixed).

**The fix**: apply the same explicit `FEATURE_COLS` list in the production retraining cell, and confirm the real shape of the ONNX graph before exporting:

```python
X = df_train[FEATURE_COLS]  # before: df_train.drop(columns=['target'])
model_ia.fit(X, y)
print(X.shape)  # confirm (n, 8), not (n, 9)
```

## The general lesson

None of these three leaks was visible by looking only at the final metrics — all three required reviewing the code cell by cell, not just the output numbers. That is why a "nice-looking" backtest result should never be accepted without auditing the full pipeline that produced it.
