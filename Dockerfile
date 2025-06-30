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
    artifacts/evaluation

# Copy initial data if it exists
RUN if [ -f "data/data_ingestion/input_raw.csv" ]; then \
    cp data/data_ingestion/input_raw.csv artifacts/data_ingestion/; \
    fi

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 