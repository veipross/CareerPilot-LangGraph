FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 careerpilot \
    && useradd --uid 10001 --gid careerpilot --create-home --shell /usr/sbin/nologin careerpilot

COPY --chown=careerpilot:careerpilot . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

RUN mkdir -p /app/outputs \
    && chown -R careerpilot:careerpilot /app/outputs

USER careerpilot

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "careerpilot.api:app", "--host", "0.0.0.0", "--port", "8001"]
