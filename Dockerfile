FROM python:3.10-slim

# Install system libraries required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    mediapipe \
    opencv-python-headless \
    numpy \
    scikit-learn \
    joblib \
    Flask \
    gunicorn

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

# Start Flask app using Gunicorn WSGI server on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "webapp.app:app"]
