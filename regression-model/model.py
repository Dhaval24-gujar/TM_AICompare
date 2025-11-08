from sklearn.linear_model import LinearRegression
from codecarbon import EmissionsTracker
import numpy as np
import time
import random
import os

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
    output_file="linear_model_emissions.csv"
)

print("🚀 Linear Regression model started. Tracking emissions continuously...\n")
tracker.start()

try:
    while True:
        val = random.uniform(0, 10)
        pred = model.predict(np.array([[val]]))[0]
        print(f"📊 Input: {val:.2f} ➡️ Prediction: {pred:.3f}")
        time.sleep(5)
        tracker.flush()  # write partial usage to log

except KeyboardInterrupt:
    print("\n🛑 Stopping linear regression model...")
    tracker.stop()
