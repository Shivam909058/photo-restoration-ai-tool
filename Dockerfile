FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Clone GFPGAN if not exists
RUN if [ ! -d "GFPGAN" ] ; then git clone https://github.com/TencentARC/GFPGAN.git ; fi

# Install GFPGAN
RUN cd GFPGAN && \
    pip install -r requirements.txt && \
    python setup.py develop && \
    cd ..

# Create necessary directories
RUN mkdir -p experiments/pretrained_models uploads results

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 