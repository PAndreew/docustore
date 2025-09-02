# api/Dockerfile

FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV HF_HOME="/root/.cache/huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="/root/.cache/torch/sentence_transformers"

RUN pip install uv

COPY pyproject.toml ./
COPY config.toml ./

RUN uv pip install --system --no-cache ".[all]"

# Copy the pre-cache script into the container
COPY precache_models.py ./
# Run the script to download the models. This layer will be cached by Docker.
RUN python precache_models.py

COPY src/ .

# Expose the port the app runs on
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]