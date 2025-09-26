FROM python:3.12-slim

ENV PYTHONDONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
  
WORKDIR /src

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .