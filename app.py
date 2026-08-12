import pickle
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils.validation import check_is_fitted
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError(
        "Missing OpenAI or OpenRouter API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY in your environment."
    )

STOP_WORDS = ENGLISH_STOP_WORDS
STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r"\b\w+\b")

model = ChatOpenAI(
    model_name="poolside/laguna-s-2.1:free",
    temperature=1,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENAI_API_KEY,
)

def reply_to_llm(text):
    res = model.invoke(f"Give the reply, firstly to understand the user query then base on the user query give the politly and formally reply to user, Constrain:- Don't mention the [Your Name] [Your Position] [Company Name] and give reply in just 2-3 line. User query: {text}")
    return res.content

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = TOKEN_PATTERN.findall(text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(STEMMER.stem(token) for token in tokens)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_vectorizer_fitted(vectorizer, corpus):
    try:
        check_is_fitted(vectorizer)
    except NotFittedError:
        vectorizer.fit(corpus)
    return vectorizer


def load_training_data():
    csv_path = Path(__file__).resolve().parent / "customer_query_dataset_humanized_12924.csv"
    df = pd.read_csv(csv_path, usecols=["Customer_query", "Category", "Intent", "Priority"])
    df["Customer_query"] = df["Customer_query"].fillna("").astype(str)
    return df


def ensure_classifier_compatible(classifier, tfidf, texts, labels, pickle_path: str = None):
    needs_retrain = False
    try:
        check_is_fitted(classifier)
    except NotFittedError:
        needs_retrain = True

    if hasattr(classifier, "n_features_in_"):
        current_features = tfidf.transform([texts[0]]).shape[1]
        if classifier.n_features_in_ != current_features:
            needs_retrain = True

    if needs_retrain:
        X = tfidf.transform(texts)
        classifier.fit(X, labels)
        if pickle_path is not None:
            with open(pickle_path, "wb") as f:
                pickle.dump(classifier, f)
    return classifier


def load_models():
    training_df = load_training_data()
    training_texts = [preprocess_text(text) for text in training_df["Customer_query"].tolist()]

    category_tfidf = ensure_vectorizer_fitted(load_pickle("category_tfidf_vectorizer.pkl"), training_texts)
    category_model = ensure_classifier_compatible(
        load_pickle("category_classifier.pkl"),
        category_tfidf,
        training_texts,
        training_df["Category"].astype(str).tolist(),
        "category_classifier.pkl",
    )

    intent_tfidf = ensure_vectorizer_fitted(load_pickle("intent_tfidf_vectorizer.pkl"), training_texts)
    intent_model = ensure_classifier_compatible(
        load_pickle("intent_classifier.pkl"),
        intent_tfidf,
        training_texts,
        training_df["Intent"].astype(str).tolist(),
        "intent_classifier.pkl",
    )

    priority_tfidf = ensure_vectorizer_fitted(load_pickle("priority_tfidf_vectorizer.pkl"), training_texts)
    priority_model = ensure_classifier_compatible(
        load_pickle("priority_classifier.pkl"),
        priority_tfidf,
        training_texts,
        training_df["Priority"].astype(str).tolist(),
        "priority_classifier.pkl",
    )

    return {
        "category": (category_tfidf, category_model),
        "intent": (intent_tfidf, intent_model),
        "priority": (priority_tfidf, priority_model),
    }


def predict_label(model_components, text: str) -> str:
    tfidf, classifier = model_components
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    return classifier.predict(vectorized_text)[0]


# ======================================================================================
# ------------------------------------ UI SECTION ---------------------------------------
# Everything below this line is presentation-only. No classification/LLM logic changed.
# ======================================================================================

st.set_page_config(
    page_title="Customer Query Classifier",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Light custom styling (CSS only, no logic) ----
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            background-color: rgba(120, 120, 120, 0.08);
            border: 1px solid rgba(120, 120, 120, 0.15);
            margin-bottom: 0.8rem;
        }
        .result-label {
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.2rem;
        }
        .result-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
        .reply-box {
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            background-color: rgba(16, 163, 127, 0.08);
            border: 1px solid rgba(16, 163, 127, 0.25);
            line-height: 1.5;
        }
        .priority-high { color: #dc2626; }
        .priority-medium { color: #d97706; }
        .priority-low { color: #16a34a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🎧 About")
    st.write(
        "This tool classifies an incoming customer query by **Category**, "
        "**Intent**, and **Priority**, and drafts a suggested reply using an LLM."
    )
    st.markdown("---")
    st.markdown("### 📝 Tips")
    st.write(
        "- Paste the query exactly as received for best accuracy.\n"
        "- Longer, more detailed queries tend to classify better.\n"
        "- The generated reply is a draft — review before sending."
    )


# ---- Header ----
st.markdown('<div class="main-title">🎧 Customer Query Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classify a customer query and generate a suggested reply in one click.</div>',
    unsafe_allow_html=True,
)

# ---- Load models (with spinner instead of a silent blocking call) ----
with st.spinner("Loading models..."):
    models = load_models()

# ---- Input area ----
input_sms = st.text_area(
    "Enter the customer query",
    height=150,
    placeholder="e.g. My order hasn't arrived yet and it's been two weeks. Can you help?",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not input_sms or not input_sms.strip():
        st.warning("Please enter a query before predicting.")
    else:
        with st.spinner("Classifying query and generating reply..."):
            category_prediction = predict_label(models["category"], input_sms)
            intent_prediction = predict_label(models["intent"], input_sms)
            priority_prediction = predict_label(models["priority"], input_sms)
            reply = reply_to_llm(input_sms)

        st.markdown("### Predictions")

        priority_class = {
            "high": "priority-high",
            "medium": "priority-medium",
            "low": "priority-low",
        }.get(str(priority_prediction).strip().lower(), "")

        pred_col1, pred_col2, pred_col3 = st.columns(3)
        with pred_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Category</div>
                    <div class="result-value">{category_prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pred_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Intent</div>
                    <div class="result-value">{intent_prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pred_col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Priority</div>
                    <div class="result-value {priority_class}">{priority_prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 💬 Suggested Reply")
        st.markdown(f'<div class="reply-box">{reply}</div>', unsafe_allow_html=True)

        with st.expander("Copy reply text"):
            st.code(reply, language=None)