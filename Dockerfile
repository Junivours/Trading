FROM python:3.11-slim

# System-Pakete installieren und TA-Lib kompilieren
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# TA-Lib Library Pfad setzen (meist nicht nötig, aber sicherer)
ENV LD_LIBRARY_PATH=/usr/local/lib

# Arbeitsverzeichnis setzen
WORKDIR /app

# Python-Requirements installieren
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Restlichen Code kopieren
COPY . .

# Startbefehl für Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
