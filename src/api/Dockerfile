# api/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv for package management
RUN pip install uv

# 1. Copy ONLY the dependency file first.
COPY pyproject.toml ./

# 2. Install dependencies. This slow step remains cached.
RUN uv pip install --system --no-cache ".[all]"

# 3. CRITICAL FIX: Copy the contents of your 'src' directory into the WORKDIR.
COPY src/ .

# Expose the port the app runs on
EXPOSE 8080

# Now this command will work because '/app/api' exists.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]