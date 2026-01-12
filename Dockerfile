FROM python:3.12-slim

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Ensure requirements.txt has a recent Streamlit (e.g., streamlit>=1.35,<2)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# map port 8501 for Streamlit default port to 8501 on host
EXPOSE 8501


ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]