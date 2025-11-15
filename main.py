from flask import Flask, request, jsonify, render_template
import joblib
import requests
import os

app = Flask(__name__)

# --------- Load ML Model -----------
model = joblib.load("sentiment_svm.joblib")

# ---------- OpenRouter API Key -------------
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------- Function: Analyze Review Using ML Model ----------
def analyze_sentiment(text):
    prediction = model.predict([text])[0]
    return prediction  # Positive or Negative


# ---------- Function: Get LLM Response ----------
def llm_response(review_text, sentiment):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",     # or any model from OpenRouter
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that generates helpful responses to customer reviews."
            },
            {
                "role": "user",
                "content": f"Review: {review_text}\nSentiment: {sentiment}\nGenerate a professional response."
            }
        ]
    }

    response = requests.post(OPENROUTER_URL, json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]


# ---------- API Endpoint ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    review = data["review"]

    sentiment = analyze_sentiment(review)
    reply = llm_response(review, sentiment)

    return jsonify({
        "sentiment": sentiment,
        "response": reply
    })


# --------- Optional Home Route -------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------- Run Server ---------
if __name__ == "__main__":
    app.run(debug=True)
