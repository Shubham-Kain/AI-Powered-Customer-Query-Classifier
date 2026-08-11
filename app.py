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

STOP_WORDS = ENGLISH_STOP_WORDS
STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r"\b\w+\b")

model = ChatOpenAI(
    model_name="poolside/laguna-s-2.1:free",
    temperature=1,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
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


st.title("Customer Query Classifier")
models = load_models()

input_sms = st.text_area("Enter the query", height=150)
if st.button("Predict"):
    category_prediction = predict_label(models["category"], input_sms)
    intent_prediction = predict_label(models["intent"], input_sms)
    priority_prediction = predict_label(models["priority"], input_sms)
    reply = reply_to_llm(input_sms)

    st.header("Predictions")
    st.write(f"Category: {category_prediction}")
    st.write(f"Intent: {intent_prediction}")
    st.write(f"Priority: {priority_prediction}")
    st.header("Reply to User")
    st.write(f"Reply: {reply}")
