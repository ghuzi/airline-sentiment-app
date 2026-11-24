# ✈️ SkyDesk: Airline Tweet Sentiment Console

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

An airline support team receives thousands of tweets every day. **SkyDesk** reads each tweet, predicts whether the customer is *Negative*, *Neutral* or *Positive*, and ranks the queue so unhappy customers get a reply first.

**Live demo:** _add the Streamlit link here after deploying_

<!-- After deploying, add screenshots to a docs/ folder and uncomment:
![Single tweet view](docs/single-tweet.png)
![Triage queue](docs/triage-queue.png)
-->

## Features

- **Single tweet:** sentiment, class probabilities and a queue priority (HIGH / STANDARD / LOW).
- **Triage queue:** paste one tweet per line and get a table ranked by negative score.
- **Transparent:** shows the cleaned text the model actually received.

## Results

Evaluated on a held-out test set of 2,880 tweets. Models were compared by **macro F1**, because the data is imbalanced (an "always Negative" guess already scores 0.631 accuracy).

| Model | Macro F1 | Accuracy |
|---|---|---|
| **Logistic Regression + TF-IDF (final)** | **0.727** | **0.780** |
| Random Forest (untuned) | 0.709 | 0.785 |
| Decision Tree (untuned) | 0.622 | 0.692 |
| VADER (rule-based, no training) | 0.449 | 0.476 |

Per-class F1 of the final model: Negative 0.861, Neutral 0.610, Positive 0.708.

- In 5-fold cross-validation, Logistic Regression beat an untuned Random Forest in **all 5 folds** (macro F1 0.724 vs 0.694).
- Hyperparameter tuning (GridSearchCV) gave no real gain over the defaults (0.728 before, 0.727 after), which suggests the limit comes from the data rather than the settings.
- VADER was run on the same test tweets as a no-training baseline. It is not a like-for-like comparison, since the final model was trained on airline tweets and VADER was not.

## Approach

1. **Data:** Twitter US Airline Sentiment (Kaggle), 14,640 tweets from 16 to 24 Feb 2015. 14,400 remained after cleaning.
2. **Data problems fixed:**
   - Over 1,500 tweets had label names inserted into the text (for example "cancelled" became "Cancelled Flightled"). These were repaired.
   - Duplicate rows and duplicate tweet IDs were removed, and texts that appeared with conflicting labels were dropped.
   - Tweets that were empty after text cleaning were dropped.
3. **Text cleaning:** lowercase, remove URLs, mentions and punctuation, expand contractions, remove stopwords, lemmatize. **Negation words (not, no, never, ...) are kept on purpose.** The same function is used in training and in this app.
4. **Model:** TF-IDF (1-2 word n-grams) + Logistic Regression with balanced class weights, saved as one scikit-learn pipeline.
5. **Three classes:** Neutral is kept because real support teams receive neutral tweets too.

## Known limits

- **Neutral is the hardest class**, because neutral and positive tweets overlap and some labels are low-confidence.
- **Negation is not truly understood.** The word "not" acts as a strong Negative cue in the model, so "The flight was not bad" is predicted Negative.
- **Narrow data:** 8 days of US airline tweets from 2015. Other airlines, regions, languages and newer slang are not covered.
- Sarcasm was not tested.
- Use it to prioritise replies, not to replace a human reading the tweet.

## Run locally

```bash
git clone https://github.com/ghuzi/airline-sentiment-app.git
cd airline-sentiment-app
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Tested with Python 3.11 and scikit-learn 1.6.1. The scikit-learn version is pinned because the saved model was trained with it.

## Project structure

```
airline-sentiment-app/
├── app.py                      # Streamlit interface + text cleaning
├── sentiment_pipeline.joblib   # trained TF-IDF + Logistic Regression pipeline
├── requirements.txt
├── .streamlit/config.toml      # theme
└── README.md
```

## Tech stack

Python, pandas, NLTK, scikit-learn, Streamlit, joblib.

## Author

Built by [@ghuzi](https://github.com/ghuzi).
