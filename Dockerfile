FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8765 \
    WARP_CACHE_PATH=/cache/warp \
    WINDTUNNEL_GEN_DIR=/cache/gen \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY web ./web

EXPOSE 8765
VOLUME /cache
CMD ["python", "server/app.py"]
