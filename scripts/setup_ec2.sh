#!/bin/bash

# Update the instance
sudo apt-get update -y
sudo apt-get upgrade -y

# Install necessary tools
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    unzip

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create directories
mkdir -p ~/sentiment-analysis/logs
mkdir -p ~/sentiment-analysis/data

# Create docker-compose.yml
cat > ~/sentiment-analysis/docker-compose.yml << 'EOF'
version: '3'

services:
  sentiment-api:
    image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest
    ports:
      - "8080:8080"
    environment:
      - AWS_REGION=ap-southeast-2
      - BUCKET_NAME=mlop
    volumes:
      - ./logs:/app/logs
    restart: always
EOF

echo "EC2 instance setup completed" 