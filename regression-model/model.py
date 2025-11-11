import threading
import boto3
import numpy as np
from sklearn.linear_model import LinearRegression
from codecarbon import EmissionsTracker
import time
import random
import os
from dotenv import load_dotenv

load_dotenv()

# --- AWS S3 Setup ---
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

BUCKET_NAME = "carbon-logs-dev"
FOLDER_NAME = "emissions"
FILE_NAME = "linear_regression_model_emissions.csv"
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
            s3.upload_file(EMISSIONS_FILE, BUCKET_NAME,f"{FOLDER_NAME}/{FILE_NAME}")
            print("✅ Uploaded emissions log to S3 (periodic)")
        except Exception as e:
            print("⚠️ Upload failed:", e)


def run_linear_regression_model():
    # Ensure log directory exists
    os.makedirs("/app/emissions_logs", exist_ok=True)

    # Dummy data and model
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.2, 4.1, 6.1, 8.0, 9.9])
    model = LinearRegression()
    model.fit(X, y)

    # Initialize tracker
    tracker = EmissionsTracker(
        project_name="LinearRegressionModel",
        output_dir="/app/emissions_logs",
        save_to_file=True,
        output_file=f"{FILE_NAME}",
        measure_power_secs=5, # more frequent logging
    )

    print("🚀 Linear Regression model started. Tracking emissions continuously...\n")
    tracker.start()

    
    while True:
        val = random.uniform(0, 10)
        pred = model.predict(np.array([[val]]))[0]
        print(f"📊 Input: {val:.2f} ➡️ Prediction: {pred:.3f}")
        time.sleep(5)
        tracker.flush()  # write partial usage to log
        

if __name__ == "__main__":
    t1 = threading.Thread(target=run_linear_regression_model)
    t2 = threading.Thread(target=upload_to_s3)
    t1.start()
    t2.start()
    t1.join()
    t2.join()