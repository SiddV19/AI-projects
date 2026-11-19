# AI Stock Market Predictor

**Python · PyTorch · LSTM · Google Colab**

[← AI projects](../README.md) · [Open in Google Colab](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2) · [View the tested notebook](stock_market_predictor.ipynb) · [Source and attribution](SOURCE.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2)

Predict Microsoft (MSFT)'s next-session adjusted closing price from the previous 50 trading sessions of High, Low, Open, Close, and Volume data. This learning project adapts the author's Harvard AI bootcamp course notebook and compares an LSTM with a previous-close baseline.

## Run it

1. Open the linked Colab notebook. Viewers can select **File → Save a copy in Drive** to edit and run their own copy.
2. Select **Runtime → Restart session and run all** so data preparation and model setup run before training.
3. Read the held-out prediction chart, baseline comparison, and final **VALIDATION PASSED** output.

The first code cell installs missing dependencies from PyPI. The notebook selects CUDA when available and otherwise uses CPU. The executed validation run used a Colab T4 GPU.

For an existing local Jupyter or VS Code notebook environment, install the observed package versions with `python -m pip install -r requirements.txt`, then open `stock_market_predictor.ipynb` and run its cells in order.

## Method

- Download one ticker from Yahoo Finance through `yfinance`, with adjusted OHLC prices and a fixed end date of **2026-11-19, exclusive**. The saved run received **4,245 observations**, ending **2026-11-17**.
- Split observations chronologically: the first 80% for training and the remaining 20% for testing. Fit the feature and target scalers **only on the training period**.
- Use 50-session input windows with five features. Every target occurs after its final input day.
- Train a two-layer LSTM with 64 hidden units, a five-unit ReLU head, and a scalar output. Its output adds a learned adjustment to the last observed scaled close.
- Use Adam, learning rate 0.001, batch size 64, 30 epochs, and seed 42. Report sample-weighted mean training MAE for each logged epoch.
- Evaluate one-step-ahead predictions on held-out dates, using only observations available before each target date. Model settings are fixed before inspecting test results.

Predictions and targets both have shape `[batch, 1]`; an assertion prevents accidental broadcasting. Inference uses `model.eval()` and `torch.no_grad()`. Inputs remain available as NumPy arrays, so creating the datasets again does not trigger the earlier tensor-conversion error.

## Saved evaluation

The complete notebook ran from a restarted Colab session on **November 18, 2026 (America/New_York)**. All 16 code cells executed without errors, and the final validation checks passed.

**Held-out period:** 2023-07-03–2026-11-17, **849 predictions**.

| Metric | LSTM | Previous-close baseline |
| --- | ---: | ---: |
| MAE (USD) | 4.7550 | 4.7607 |
| RMSE (USD) | 6.9215 | 6.9238 |
| MAPE (%) | 1.1305 | 1.1318 |
| MDAPE (%) | 0.8236 | 0.8232 |

The LSTM reduced test MAE by approximately **0.12%** relative to the baseline: the two approaches essentially matched. This single holdout result does not establish a useful predictive edge. The experiment does not evaluate trading returns or transaction costs, and no confidence interval is estimated.

The final inference cell uses the evaluated training-only model and the most recent observed 50-session window. It labels its output as an experimental next-session forecast and reports the actual last-observed date.

## Files

| File | Purpose |
| --- | --- |
| [stock_market_predictor.ipynb](stock_market_predictor.ipynb) | Complete tested notebook, including saved charts and text outputs |
| [training_excerpt.py](training_excerpt.py) | The notebook's training/inference cell; requires earlier setup cells |
| [requirements.txt](requirements.txt) | Package versions observed in the successful Colab run; PyTorch's CUDA suffix is omitted for package installation |
| [validation_results.json](validation_results.json) | Test metrics, dates, configuration, package versions, checks, and a code-source SHA-256 |
| [SOURCE.md](SOURCE.md) | Course context, attribution, and changes |

## Validation and reproducibility

The notebook checks target shapes during training, alignment of held-out dates and prices, the number of observations used to fit each scaler, and finite losses and predictions. A separate synthetic-data check verified that a large price change present only in the test period cannot affect the fitted scaler and that each input window excludes its target day.

Yahoo Finance can revise historical adjusted prices, and results may vary across package versions and hardware despite the fixed seed. The committed notebook is a tested snapshot of the linked Colab notebook. Future edits to either copy do not synchronize automatically.

Relevant documentation: [PyTorch L1Loss shapes](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.L1Loss.html), [scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), and [yfinance download parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
