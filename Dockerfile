
# Start with Python 3.13
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# The FastAPI application listens on port 8000
EXPOSE 8000

# Start the travel API
CMD ["uvicorn", "day19_travelapi:app", "--host", "0.0.0.0", "--port", "8000"]
