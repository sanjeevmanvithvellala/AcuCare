FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create raw/processed data folders and log folder
RUN mkdir -p data/raw data/processed models/registry logs reports/plots reports/pdfs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default command is running the backend API
CMD ["python", "-m", "app.backend.main"]
