# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application scripts
COPY scripts /app/scripts
COPY models/tokenizer.model /app/models/tokenizer.model

# Create a non-root user with the same UID/GID as the host user
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} appgroup && \
    useradd -u ${USER_UID} -g ${USER_GID} -m appuser

# Switch to the non-root user
USER appuser

# Set the default command to run the application
CMD ["python", "scripts/main.py"]
