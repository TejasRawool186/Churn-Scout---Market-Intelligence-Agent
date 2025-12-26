# Use Apify's optimized Python image
FROM apify/actor-python:3.11

# 1. Install Playwright & Browsers
RUN pip install playwright
RUN playwright install --with-deps chromium

# 2. Install Project Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Source Code
COPY . .

# 4. Entry Point
CMD ["python3", "-m", "src.main"]
