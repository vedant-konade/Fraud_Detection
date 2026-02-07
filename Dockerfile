FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .
COPY models /app/models

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
