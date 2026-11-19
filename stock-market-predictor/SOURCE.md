# Source and attribution

- **Repository maintainer:** Siddharth Vadali.
- **Course context:** The original notebook supplied by the author is labeled “Harvard AI Bootcamp.” Its course heading and notebook structure are preserved.
- **Original Colab file:** [AI Stock Market Predictor](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2). The existing Drive file ID is retained.
- **Complete tested snapshot:** [stock_market_predictor.ipynb](stock_market_predictor.ipynb), executed in a restarted Colab session with saved outputs.
- **Training excerpt:** [training_excerpt.py](training_excerpt.py) matches the training/inference cell in that snapshot and depends on the earlier notebook cells.

The project adapts course-provided starter material. Preserve any course, instructor, collaborator, and starter-code attribution when reusing or extending it.

## Changes in the tested version

- Normalize single-ticker Yahoo Finance columns and identify the selected ticker correctly as Microsoft (MSFT).
- Split data chronologically before fitting feature or target scalers.
- Keep predictions and targets shaped as `[batch, 1]` and make dataset tensor conversion safe to repeat.
- Use an LSTM adjustment to the previous close, with initialization at the persistence baseline.
- Handle CPU/CUDA placement, evaluation mode, inference without gradients, and mean epoch loss.
- Fix plot coverage, metric names, target/date alignment, and percentage-change calculation.
- Add a previous-close benchmark, end-to-end checks, recorded package versions, and an evaluation record.

The original incorrect saved outputs were cleared before the clean run. The reported results in the notebook and README come from the corrected run. Later edits in Colab or GitHub require another explicit synchronization.
