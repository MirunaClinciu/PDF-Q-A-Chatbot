# Use Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies required for sentence-transformers & PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir faiss-cpu  # <-- Add this line

# Copy app code
COPY . .

# Expose port
EXPOSE 5000

# Start the Flask app via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers=1", "--threads=1", "--timeout=180"]

