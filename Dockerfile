FROM python:3.11-slim AS base_image

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

FROM base_image AS download_model

WORKDIR /app

ARG TARGETARCH
ARG MODEL_NAME
ARG ONNX_RUNTIME
ENV ONNX_CPU=${TARGETARCH}
ARG TRUST_REMOTE_CODE
ARG USE_SENTENCE_TRANSFORMERS_VECTORIZER
RUN mkdir nltk_data
COPY download.py .
RUN ./download.py

FROM base_image AS t2v_transformers

WORKDIR /app
COPY --from=download_model /app/models /app/models
COPY --from=download_model /app/nltk_data /app/nltk_data
COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]
