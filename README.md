# AI Projects

Artificial intelligence and machine-learning projects by **Siddharth Vadali**, built with Python and PyTorch.

## Projects

| Project | Technologies | Available source |
| --- | --- | --- |
| **[AI Stock Market Predictor](stock-market-predictor/README.md)** | Python · PyTorch · LSTM · Google Colab | Complete executed notebook, dependencies, and evaluation record |

### AI Stock Market Predictor

A Microsoft (MSFT) next-session adjusted-close forecasting learning project adapted from the author's Harvard AI bootcamp notebook. It uses 50-session OHLCV windows, a PyTorch LSTM, chronological train/test evaluation, and scalers fitted only to training observations.

[Explore the project](stock-market-predictor/README.md) · [Open in Google Colab](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2) · [View the complete notebook](stock-market-predictor/stock_market_predictor.ipynb) · [Source and attribution](stock-market-predictor/SOURCE.md)

## Project status

The complete notebook was executed from a restarted Colab session and its validation checks passed. On 849 held-out predictions, its MAE was $4.7550, compared with $4.7607 for the previous-close baseline. These results essentially match the baseline; they do not establish a useful predictive edge.

The repository contains a tested snapshot of the linked Colab notebook, its saved outputs, dependencies, and an evaluation record. Future edits to the GitHub and Colab copies do not synchronize automatically.

## Other work

[React Native Application Portfolio](https://github.com/SiddV19/expo-app-portfolio) — mobile apps exploring sensors, GPS, drawing, games, and everyday utilities.
