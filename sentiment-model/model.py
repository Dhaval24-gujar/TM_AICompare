from transformers import pipeline
from codecarbon import EmissionsTracker
import time
import random
import os

# Ensure log directory exists
os.makedirs("/app/emissions_logs", exist_ok=True)

# Initialize model and emissions tracker
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

tracker = EmissionsTracker(
    project_name="SentimentModel",
    output_dir="/app/emissions_logs",
    save_to_file=True,
    output_file="sentiment_model_emissions.csv"
)

texts = [
    "Tech Mahindra is doing great work in AI sustainability!",
    "The product performance is disappointing and inefficient.",
    "I'm impressed with the new update and its improvements.",
    "Energy consumption of this system is too high.",
    "Our AI infrastructure is becoming more efficient."
]

print("🚀 Sentiment model started. Tracking emissions continuously...\n")
tracker.start()

try:
    while True:
        text = random.choice(texts)
        result = sentiment_pipe(text)[0]
        print(f"🧩 Input: {text}")
        print(f"➡️ Sentiment: {result['label']} (score: {result['score']:.3f})\n")
        time.sleep(5)
        tracker.flush()  # force CodeCarbon to write interim results

except KeyboardInterrupt:
    print("\n🛑 Stopping sentiment model...")
    tracker.stop()
