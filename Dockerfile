# app/Dockerfile

FROM python:3.9-slim

WORKDIR /src

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/leporejoseph/crewBot.git .

RUN pip3 install -r requirements.txt

EXPOSE 8008

HEALTHCHECK CMD curl --fail http://localhost:8008/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8008", "--server.address=0.0.0.0"]