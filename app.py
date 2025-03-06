from transformers import pipeline
import gradio as gr

# Load the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    result = sentiment_model(text)[0]  # Get first result
    return f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}"

# Create Gradio interface
interface = gr.Interface(fn=analyze_sentiment, inputs="text", outputs="text")

# Launch Gradio app (required for Hugging Face Spaces)
if __name__ == "__main__":
    interface.launch()
