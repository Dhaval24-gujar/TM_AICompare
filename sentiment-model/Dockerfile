# Base image with Python 3.13
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

RUN pip --no-cache-dir install transformers torch codecarbon boto3 dotenv

# Run your model
CMD ["python", "model.py"]