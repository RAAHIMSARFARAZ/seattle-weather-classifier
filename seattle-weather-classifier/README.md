# Seattle Weather Classification

Predicting the day's weather class - `sun`, `rain`, `fog`, `drizzle`, `snow` -
from four years of daily Seattle observations. Final project for a data science
course at NED Academy.

## The result that matters

| Split | Accuracy | Macro F1 |
|---|---|---|
| Random (`shuffle=True`, the default) | 84.6% | 0.53 |
| **Chronological (last 20% held out)** | **85.7%** | **0.38** |

Accuracy *rose* and macro F1 *halved* when the split was made honest. That
apparent contradiction is the point of this notebook.

The dataset is a time series. A random split lets the model train on 15 March
2015 and be tested on 12 March 2015 - neighbouring days with near-identical
weather. It scores well by interpolation and never has to extrapolate.

Under a chronological split the model has to predict the *future* from the past,
which is what a weather model actually does. Accuracy stays high because the
classes are imbalanced and `sun` dominates; macro F1 - the unweighted mean over
classes - collapses because the rare classes fall apart:

- **every** drizzle day predicted as sun
- **30 of 31** fog days predicted as sun
- **1 of 110** rain days missed

The model learned "it is sunny unless it visibly rains" and has no idea what fog
or drizzle look like.

## What is in the notebook

1. Load and inspect (~1,400 daily rows: precipitation, max/min temperature,
   wind, weather label).
2. Clean: drop the unneeded `date`-derived columns, handle the `NaN`s in
   `precipitation`, map string labels to integers.
3. Scale features and compare classifiers - k-Nearest Neighbours, decision tree,
   random forest, logistic regression.
4. Evaluate: accuracy, macro F1, confusion matrices.
5. **The split fix**: sort by date, then `train_test_split(..., shuffle=False)`,
   and re-evaluate everything.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook seattle-weather-classifier.ipynb
```

The data file is `seattle-weather.csv` (public, from the Seattle weather
dataset). Update the `read_csv` path on the first code cell.

## What I would do next

- **Merge the rare classes.** 31 fog days and 4 drizzle days in four years
  cannot support a five-class model. `sun / rain / other` would be honest, or
  treat fog and drizzle as separate binary detectors.
- **Add lags and rolling means.** Yesterday's temperature and a 3-day
  precipitation total are the features a real forecast uses. The model
  currently sees only today's readings, which borders on leakage in the other
  direction - precipitation today is almost a restatement of the label.
- **Report precision/recall per class**, not just macro F1, so the fog failure
  is visible rather than averaged away.
- **Class weights or resampling** for the rare classes.

## Honest limitations

- **Leakage in the features.** `precipitation` and `wind` are measured on the
  same day as the label. Predicting the label from them is closer to
  classification than forecasting; a genuine forecast would use only prior days.
- **Small, single-city sample.** Four years of one station. Nothing learned here
  transfers to another climate without revalidation.
- **No snow in the test period**, so the snow row of the confusion matrix is
  empty - the model was never actually tested on it.

## Figures

| File | Shows |
|---|---|
| `figures/model-comparison.png` | Accuracy and F1 across the candidate classifiers |
| `figures/confusion-matrix-chronological.png` | The honest split - where fog and drizzle collapse |
| `figures/confusion-matrix-knn.png` | The same matrix for k-NN, for contrast |
