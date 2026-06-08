FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for websocket-client compilation if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application modules
COPY main.py .
COPY explore_data.py .
COPY control_api.py .
COPY config.py .
COPY models.py .
COPY data_fetcher.py .
COPY market_manager.py .
COPY alert_manager.py .
COPY wallet_cache.py .
COPY detectors/ ./detectors/
COPY clustering/ ./clustering/

# Create directory for database (will be mounted as volume)
RUN mkdir -p /app/data

# Run the detector
CMD ["python", "-u", "main.py"]
