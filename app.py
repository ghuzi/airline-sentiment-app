import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="SkyDesk | Customer Voice Console", page_icon="✈️", layout="centered")

LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#C8102E", "Neutral": "#5B6B7F", "Positive": "#0B7A4B"}
PRIORITY = {"Negative": ("HIGH", "Reply first"),
            "Neutral": ("STANDARD", "Normal queue"),
            "Positive": ("LOW", "No action needed")}


def html(s):
    """Render HTML; strip indentation and blank lines so Markdown does not treat it as code."""
    st.markdown(re.sub(r"\n\s+", "\n", s.strip()), unsafe_allow_html=True)


# ---------- Model and cleaning (same logic as the notebook) ----------
@st.cache_resource
def load_resources():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    model = joblib.load(Path(__file__).parent / "sentiment_pipeline.joblib")
    negations = {"no", "nor", "not", "never", "neither", "none", "nobody", "nothing", "nowhere"}
    stop_words = set(stopwords.words("english")) - negations
    stop_words -= {"didn", "doesn", "don", "isn", "wasn", "weren", "aren", "hasn", "haven",
                   "hadn", "couldn", "wouldn", "shouldn", "needn", "mustn", "mightn",
                   "shan", "won", "ain"}
    return model, stop_words, WordNetLemmatizer()


model, STOP_WORDS, LEMMATIZER = load_resources()
SPECIAL = {"can't": "can not", "won't": "will not", "cannot": "can not", "ain't": "is not"}


def expand_contractions(text):
    text = text.replace("’", "'")
    for k, v in SPECIAL.items():
        text = text.replace(k, v)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'m\b", " am", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'d\b", " would", text)
    return re.sub(r"'s\b", "", text)


def clean_text(text):
    text = expand_contractions(text.lower())
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(t, "v"), "n") for t in tokens]
    return " ".join(tokens)


# ---------- Styling ----------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 860px; }
header[data-testid="stHeader"] { display: none !important; }
.stApp, [data-testid="stMain"], [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
.topbar { position: relative; }
.topbar { background: #0B2545; margin: 0 -9999px 28px; padding: 16px 9999px; display: flex; align-items: center; gap: 12px; }
.topbar .logo { width: 36px; height: 36px; border-radius: 8px; background: #13A0E0; display: flex; align-items: center; justify-content: center; }
.topbar .brand { color: #fff; font-weight: 800; letter-spacing: 0.14em; font-size: 17px; }
.topbar .sub { color: #9DB4D0; font-size: 13px; margin-left: auto; }
.headline { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 4px; color: #0B2545; }
.lede { color: #5B6B7F; font-size: 15px; margin-bottom: 18px; }
.eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 0.14em; color: #7A8CA3; text-transform: uppercase; }
.pass { display: flex; background: #fff; border: 1px solid #DCE3EC; border-radius: 14px; box-shadow: 0 6px 20px rgba(11,37,69,0.07); margin: 18px 0 8px; overflow: hidden; }
.pass-main { flex: 1; padding: 22px 26px; }
.pass-stub { width: 190px; padding: 22px 20px; border-left: 2px dashed #CBD5E1; background: #F8FAFC; display: flex; flex-direction: column; justify-content: center; }
.verdict { font-size: 38px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; margin: 4px 0 2px; }
.conf { color: #5B6B7F; font-size: 14px; margin-bottom: 16px; }
.bar-row { display: flex; align-items: center; gap: 12px; margin: 7px 0; font-size: 13px; }
.bar-row .name { width: 70px; color: #3B4A5E; font-weight: 600; }
.bar-row .track { flex: 1; height: 8px; background: #E8EDF3; border-radius: 4px; overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 4px; }
.bar-row .pct { width: 48px; text-align: right; color: #3B4A5E; font-variant-numeric: tabular-nums; }
.prio { font-size: 26px; font-weight: 800; letter-spacing: 0.04em; margin: 4px 0; }
.note { font-size: 13px; color: #5B6B7F; }
.footer { margin-top: 36px; padding-top: 14px; border-top: 1px solid #DCE3EC; color: #7A8CA3; font-size: 12px; line-height: 1.7; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
[data-testid="stTextArea"] textarea { border-radius: 10px; font-size: 15px; }
.stButton > button { border-radius: 8px; font-weight: 600; }
@media (max-width: 640px) { .pass { flex-direction: column; } .pass-stub { width: auto; border-left: 0; border-top: 2px dashed #CBD5E1; } .topbar .sub { display: none; } }
</style>
<div class="topbar">
<div class="logo"><svg viewBox="0 0 24 24" width="22" height="22"><path fill="white" d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg></div>
<div class="brand">SKYDESK</div>
<div class="sub">Customer Voice Console</div>
</div>
""")

html("""
<div class="headline">Who needs a reply first?</div>
<div class="lede">Paste customer tweets and the model flags the unhappy ones so your team can answer them first.</div>
""")

EXAMPLES = {
    "Lost baggage": "Worst airline ever, lost my bags and nobody helped me",
    "Crew praise": "The crew was amazing, thank you so much!",
    "Flight info": "My flight is at 6 pm tomorrow from Islamabad",
    "Cancellation": "I can't believe they cancelled my flight again",
}
if "tweet" not in st.session_state:
    st.session_state["tweet"] = ""


def set_example(text):
    st.session_state["tweet"] = text


def bars_html(probs):
    rows = ""
    for label, p in zip(LABELS, probs):
        rows += (f'<div class="bar-row"><div class="name">{label}</div>'
                 f'<div class="track"><div class="fill" style="width:{p*100:.1f}%;background:{COLORS[label]}"></div></div>'
                 f'<div class="pct">{p:.0%}</div></div>')
    return rows


tab_single, tab_queue = st.tabs(["Single tweet", "Triage queue"])

# ---------- Tab 1: single tweet ----------
with tab_single:
    st.text_area("Customer tweet", key="tweet", height=110,
                 placeholder="e.g. @AirlineName my flight was delayed for 5 hours and nobody told us why...")
    st.caption("Sample tweets")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    tweet = st.session_state["tweet"].strip()
    if tweet:
        cleaned = clean_text(tweet)
        if not cleaned.strip():
            st.warning("Nothing left after cleaning (only links, mentions or symbols). Try a longer tweet.")
        else:
            probs = model.predict_proba([cleaned])[0]
            pred = LABELS[int(probs.argmax())]
            prio, note = PRIORITY[pred]
            html(f"""
            <div class="pass">
            <div class="pass-main">
            <div class="eyebrow">Predicted sentiment</div>
            <div class="verdict" style="color:{COLORS[pred]}">{pred}</div>
            <div class="conf">Model confidence {probs.max():.0%}</div>
            {bars_html(probs)}
            </div>
            <div class="pass-stub">
            <div class="eyebrow">Queue priority</div>
            <div class="prio" style="color:{COLORS[pred]}">{prio}</div>
            <div class="note">{note}</div>
            </div>
            </div>
            """)
            with st.expander("Model input (cleaned text)"):
                st.code(cleaned)

# ---------- Tab 2: triage queue ----------
with tab_queue:
    st.write("Paste one tweet per line. The queue is ranked by how likely each tweet is to be negative.")
    default_queue = "\n".join([
        "Stuck on the tarmac for 3 hours and the crew will not tell us anything",
        "Thanks for rebooking me so quickly, great service",
        "What time does the Lahore flight board tomorrow?",
        "Lost my luggage again, this is the third time this year",
        "The seats were not comfortable but the staff were kind",
        "Can you share the baggage allowance for international flights?",
    ])
    queue_text = st.text_area("Tweets (one per line)", value=default_queue, height=180, key="queue_text")
    if st.button("Rank queue", type="primary"):
        lines = [l.strip() for l in queue_text.splitlines() if l.strip()][:200]
        rows = []
        for line in lines:
            c = clean_text(line)
            if not c.strip():
                continue
            p = model.predict_proba([c])[0]
            pred = LABELS[int(p.argmax())]
            rows.append({"Priority": PRIORITY[pred][0], "Sentiment": pred,
                         "Negative score (%)": round(float(p[0]) * 100, 1), "Tweet": line})
        if rows:
            df = pd.DataFrame(rows).sort_values("Negative score (%)", ascending=False).reset_index(drop=True)
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={"Negative score (%)": st.column_config.ProgressColumn(
                    "Negative score (%)", min_value=0, max_value=100, format="%.0f")},
            )
            st.caption(f"{(df['Sentiment'] == 'Negative').sum()} of {len(df)} tweets flagged HIGH priority.")
        else:
            st.warning("No usable tweets found.")

# ---------- Footer ----------
html("""
<div class="footer">
<b>Model:</b> TF-IDF (1-2 word n-grams) + Logistic Regression, trained on 14,400 US airline tweets (Feb 2015). Test macro F1 0.727, accuracy 0.780.<br>
<b>Known limits:</b> Neutral is the hardest class. The model does not truly understand negation ("not bad" can come out Negative). Trained on 8 days of US airline tweets only. Sarcasm was not tested. Use it to prioritise replies, not to replace a human reading the tweet.
</div>
""")