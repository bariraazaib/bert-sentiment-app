%%writefile app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

st.title("Customer Feedback Sentiment Classifier (BERT)")

@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier

classifier = load_model()

user_input = st.text_area("Enter customer feedback here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        results = classifier(user_input)[0]
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        predicted_label = labels[scores.index(max(scores))]
        st.write(f"**Predicted Sentiment:** {predicted_label}")
        st.bar_chart({labels[i]: scores[i] for i in range(len(labels))})
    else:
        st.warning("Please enter some feedback text.")
