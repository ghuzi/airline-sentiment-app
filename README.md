# SkyDesk: Airline Tweet Sentiment Console

An airline support team receives thousands of tweets. This app flags the **negative** ones so unhappy customers get a reply first.

**Live demo:** _(add the Streamlit link after deploying)_

## What it does
- **Single tweet:** predicts Negative / Neutral / Positive with class probabilities and a queue priority (HIGH / STANDARD / LOW).
- **Triage queue:** paste one tweet per line and get a table ranked by negative score.

## Model
- Data: Twitter US Airline Sentiment (Kaggle), 14,400 tweets after cleaning (16 to 24 Feb 2015).
- Pipeline: text cleaning (negation words kept on purpose) + TF-IDF (1-2 word n-grams) + Logistic Regression with balanced class weights.
- Chosen by macro F1 because the data is imbalanced (always-Negative baseline accuracy is 0.631).

| Result (test set, 2,880 tweets) | Macro F1 | Accuracy |
|---|---|---|
| Logistic Regression (final) | 0.727 | 0.780 |
| VADER (no training) | 0.449 | 0.476 |

Logistic Regression also beat an untuned Random Forest in all 5 cross-validation folds (macro F1 0.724 vs 0.694).

## Known limits
- Neutral is the hardest class (neutral and positive tweets overlap).
- The model does not truly understand negation: "not" acts as a strong Negative cue, so "not bad" can come out Negative.
- Trained on 8 days of US airline tweets only; sarcasm was not tested.
- Use it to prioritise replies, not to replace a human reading the tweet.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```
Tested with Python 3.11 and scikit-learn 1.6.1.

## Files
- `app.py`: Streamlit interface and the same text-cleaning function used in training
- `sentiment_pipeline.joblib`: saved TF-IDF + Logistic Regression pipeline
- `requirements.txt`: dependencies (scikit-learn is pinned to the training version)