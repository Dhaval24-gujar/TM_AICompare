# Base image with Python 3.13
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install core dependencies
# torch-cpu avoids large CUDA downloads if you don't need GPU inside container
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir codecarbon numpy pandas scikit-learn

# Run your model
CMD ["python", "mnist.py"]
