FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    automake \
    libtool \
    pkg-config \
    && git clone https://github.com/TA-Lib/ta-lib.git \
    && cd ta-lib \
    && chmod +x autogen.sh \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CPATH=/usr/local/include

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
