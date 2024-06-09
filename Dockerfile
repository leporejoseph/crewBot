# Base stage for dependencies
FROM python:3.10-slim as base
WORKDIR /app

# Define build arguments for environment setup
ARG SKIP_DEPENDENCIES=false

# Install necessary packages, always include git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*;

# Source stage to pull latest code
FROM base as source
ARG GITHUB_TOKEN
RUN git clone https://leporejoseph:$GITHUB_TOKEN@github.com/leporejoseph/crewBot.git ./
RUN cp -a src/. ./

# Production stage for running the app
FROM base as production
COPY --from=source /app /app
RUN if [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

EXPOSE 8008
HEALTHCHECK CMD curl --fail http://localhost:8008/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8008", "--server.address=0.0.0.0"]
