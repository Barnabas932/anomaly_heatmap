FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Frontend build ----
# Node telepítés
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get update && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

COPY frontend /app/frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# ---- Backend ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

# HF Spaces default: 7860
EXPOSE 7860

CMD ["python", "-m", "backend.app"]
