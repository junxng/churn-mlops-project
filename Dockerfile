FROM python:3.11-slim

WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directory structure
RUN mkdir -p artifacts/data_ingestion \
    artifacts/data_version \
    artifacts/model_version \
    artifacts/evaluation \ 
    plots

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Expose the port the app runs on
EXPOSE 8888

# Default command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"] 