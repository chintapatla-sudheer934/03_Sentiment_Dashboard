import streamlit as st
import pandas as pd
from transformers import pipeline

st.title("Social Media Sentiment Analyzer")
st.write("Analyze a sentence or a CSV containing a `text` column.")
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")
model = load_model()
text = st.text_area("Enter a comment")
if st.button("Analyze comment") and text.strip():
    result = model(text)[0]
    st.metric("Sentiment", result["label"])
    st.metric("Confidence", f"{result['score']:.1%}")
file = st.file_uploader("Or upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    if "text" not in df.columns:
        st.error("CSV must contain a text column.")
    else:
        df["sentiment"] = df["text"].fillna("").astype(str).apply(lambda x: model(x[:512])[0]["label"] if x.strip() else "EMPTY")
        st.dataframe(df)
        st.bar_chart(df["sentiment"].value_counts())
