FROM python:3.8-slim

USER root

RUN mkdir /app
COPY . /app/
WORKDIR /app/

# Install system dependencies for xmlsec & AWS CLI
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    pkg-config \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Set Airflow configuration environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Allow script execution
RUN chmod 777 start.sh

# Set entrypoint to start Airflow
ENTRYPOINT ["/bin/sh"]
CMD ["start.sh"]
