import os
import time
import random
import threading
import boto3
from transformers import pipeline
from codecarbon import EmissionsTracker

# --- AWS S3 Setup ---
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

BUCKET_NAME = "carbon-logs-dev"
FOLDER_NAME = "emissions"
FILE_NAME = "sentiment_analysis_model_emissions.csv"
EMISSIONS_FILE = f"/app/emissions_logs/{FILE_NAME}"


def upload_to_s3():
    """Wait for CodeCarbon log file to appear, then upload immediately and every 5 min."""
    print("🕒 Waiting for emissions file to appear...")
    while not os.path.exists(EMISSIONS_FILE):
        time.sleep(2)
        print("📂 File detected! Uploading initial version to S3...")
        try:
            s3.upload_file(EMISSIONS_FILE, BUCKET_NAME, f"{FOLDER_NAME}/{FILE_NAME}")
            print("✅ Initial upload successful.")
        except Exception as e:
            print("⚠️ Initial upload failed:", e)

    # Continue periodic uploads
    while True:
        time.sleep(30)
        try:
            s3.upload_file(EMISSIONS_FILE, BUCKET_NAME, f"{FOLDER_NAME}/{FILE_NAME}")
            print("✅ Uploaded emissions log to S3 (periodic)")
        except Exception as e:
            print("⚠️ Upload failed:", e)




def run_sentiment_model():
    """Run sentiment model continuously and track emissions."""
    os.makedirs("/app/emissions_logs", exist_ok=True)
    tracker = EmissionsTracker(
        project_name="SentimentModel",
        output_dir="/app/emissions_logs",
        output_file=f"{FILE_NAME}",
        save_to_file=True,
        measure_power_secs=5, # more frequent logging
        save_to_prometheus=True,
        prometheus_url="http://host.docker.internal:9091",
    )


    tracker.start()

    sentiment = pipeline("sentiment-analysis")
    try:
        while True:
            result = sentiment("I love building AI agents!")
            print(result)
            time.sleep(random.randint(2, 5))
    except KeyboardInterrupt:
        print("\n🛑 Stopping Sentiment analysis model...")
        tracker.stop()
    finally:
        tracker.stop()
        print("🧾 Emissions tracking stopped.")


if __name__ == "__main__":
    t1 = threading.Thread(target=run_sentiment_model)
    t2 = threading.Thread(target=upload_to_s3)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
