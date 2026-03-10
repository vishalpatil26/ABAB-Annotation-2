FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir openai-whisper && \
    pip install --no-cache-dir pyannote.audio==3.1.1 && \
    pip install --no-cache-dir "gradio==3.50.2" && \
    pip install --no-cache-dir pydantic numpy==1.26.4 soundfile librosa

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
