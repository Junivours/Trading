FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Symlink für TA-Lib (damit pyta-lib sie findet)
    && ln -s /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so || true \
    && ln -s /usr/local/lib/libta_lib.a /usr/local/lib/libta-lib.a || true

ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
