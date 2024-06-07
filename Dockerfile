# app/Dockerfile

# Use a specific version of Python as the base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY src/ src/

# Install Python dependencies
RUN pip3 install -r src/requirements.txt

# Expose the port your app runs on
EXPOSE 8008

# Healthcheck for the service
HEALTHCHECK CMD curl --fail http://localhost:8008/_stcore/health

# Command to run the application
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8008", "--server.address=0.0.0.0"]
