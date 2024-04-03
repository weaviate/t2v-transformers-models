FROM --platform=$BUILDPLATFORM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG TARGETARCH
ARG MODEL_NAME
ARG ONNX_RUNTIME
ENV ONNX_CPU=${TARGETARCH}
COPY download.py .
RUN ./download.py

COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]
