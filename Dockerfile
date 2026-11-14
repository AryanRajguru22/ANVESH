# ANVESH - officer screen + API, CPU build.
#
# HONEST NOTE: the CPU image runs the same code but detection is much slower than on the NVIDIA GPU the project was developed on
# (expect a few frames per second, not real time). For a GPU host build FROM an nvidia/cuda or pytorch/pytorch image instead and install the
# CUDA torch wheels (see requirements-lock.txt). The container serves plain HTTP: put it behind an HTTPS proxy and set ANVESH_SECURE_COOKIES=1.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU torch first so ultralytics does not pull a CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY anvesh ./anvesh
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts
COPY demo/camera_config.json ./demo/camera_config.json
# detector weights: models/yolov8n.pt is NOT in git (see README) - it must exist in the build context
COPY models/yolov8n.pt ./models/yolov8n.pt
COPY models/plates ./models/plates

ENV ANVESH_HOST=0.0.0.0 ANVESH_PORT=8750 ANVESH_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8750
# first start creates an 'admin' user; the password is written to /data/initial_credentials.txt
CMD ["python", "scripts/run_app.py", "--simple-home"]
