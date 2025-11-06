# ================================
# Stage 1: Base Environment
# ================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ================================
# Stage 2: System Dependencies
# ================================
# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Stage 3: Install Python Packages
# ================================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install -U sentence-transformers

# ================================
# Stage 4: Copy Application Code
# ================================
COPY . .

# ================================
# Stage 5: Run Application
# ================================
EXPOSE 7888

# Start FastAPI using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7888"]
