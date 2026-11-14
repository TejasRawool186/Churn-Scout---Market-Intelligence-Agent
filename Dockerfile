# Use Apify's official lightweight Python 3.11 base image
FROM apify/actor-python:3.11

# Set environment variables for clean Python execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Project Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code and assets
COPY . .

# Entry Point for Apify Actor
CMD ["python3", "-m", "src.main"]
