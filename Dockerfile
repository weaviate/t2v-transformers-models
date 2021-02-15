FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY download.py .
RUN ./download.py

COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]
