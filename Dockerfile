# Use a specific version of Python as the base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Define build arguments for environment setup
ARG SKIP_DEPENDENCIES=false
ARG GITHUB_TOKEN

# Install necessary packages if not skipping dependencies
RUN if [ "$SKIP_DEPENDENCIES" = "false" ]; then \
        apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Clone the repository using the provided GitHub token
RUN git clone https://leporejoseph:$GITHUB_TOKEN@github.com/leporejoseph/crewBot.git ./

# Copy contents of src to the /app directory
RUN cp -a src/. ./

# Install Python dependencies if not skipping dependencies
RUN if [ "$SKIP_DEPENDENCIES" = "false" ]; then \
        pip install --no-cache-dir -r src/requirements.txt; \
    fi

# Expose the port your app runs on
EXPOSE 8008

# Healthcheck for the service
HEALTHCHECK CMD curl --fail http://localhost:8008/_stcore/health || exit 1

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8008", "--server.address=0.0.0.0"]
