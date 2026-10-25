FROM python:3.10-slim

# Install system libraries required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user according to Hugging Face Spaces requirements (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Install Python dependencies first for caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files with proper ownership
COPY --chown=user:user . $HOME/app

# Ensure uploads directory exists and is writable
RUN mkdir -p $HOME/app/webapp/uploads && chmod 777 $HOME/app/webapp/uploads

# Hugging Face Spaces default port
EXPOSE 7860

# Run Flask app with Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "180", "webapp.app:app"]
