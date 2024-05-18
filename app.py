import streamlit as st
import pickle
import pandas as pd
from textblob import TextBlob

# Load the pickled DataFrame with predicted sentiments
df_final = pd.read_pickle("rf_model.pkl")

# Streamlit app
def predict_sentiment(sentence):
    blob = TextBlob(sentence)
    predicted_sentiment = 'positive' if blob.sentiment.polarity > 0 else ('neutral' if blob.sentiment.polarity == 0 else 'negative')
    return predicted_sentiment, blob.sentiment.polarity

def main():
    st.title("Sentiment Analysis App")
    st.markdown("### Analyze the sentiment of your text! ")
    st.markdown("**Sentiment polarity scores:**")
    st.markdown("- *-1 to -0.5:* Very negative")
    st.markdown("- *-0.5 to -0.1:* Negative")
    st.markdown("- *-0.1 to 0.1:* Neutral")
    st.markdown("- *0.1 to 0.5:* Positive")
    st.markdown("- *0.5 to 1:* Very positive")
    # User input
    user_input = st.text_area("Enter a sentence:", height=100)
    if not user_input:
        st.warning("Please enter a sentence.")
        st.stop()

    # Make prediction
    predicted_sentiment, sentiment_polarity = predict_sentiment(user_input)

    # Display result with symbols
    st.subheader("Prediction:")
    st.write(f"Sentiment: {predicted_sentiment}")
    st.write(f"Sentiment Polarity: {sentiment_polarity}")

    # Style the result based on sentiment
    if predicted_sentiment == 'positive':
        st.success("😃 Positive sentiment")
    elif predicted_sentiment == 'neutral':
        st.info("😐 Neutral sentiment")
    else:
        st.error("😟 Negative sentiment")

if __name__ == "__main__":
    main()