FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install dvc dvc[s3] boto3 mlflow dagshub fastapi uvicorn chardet openpyxl xlrd

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV AWS_REGION=ap-southeast-2
ENV BUCKET_NAME=mlop

# Set up DVC with S3
RUN mkdir -p /root/.aws
RUN echo "[default]\nregion=${AWS_REGION}" > /root/.aws/config

# Pull model artifacts if not present
RUN dvc pull || echo "DVC pull failed, continuing..."

# Create directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/src/API

# Expose port
EXPOSE 8080

# Run the service with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
