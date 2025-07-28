FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && git clone https://github.com/TA-Lib/ta-lib.git \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so || true \
    && ln -s /usr/local/lib/libta_lib.a /usr/local/lib/libta-lib.a || true

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
