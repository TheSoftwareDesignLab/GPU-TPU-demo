FROM python:3.9-slim

WORKDIR /app

# Install FastAPI, Uvicorn, WebSockets, and ZeroMQ
RUN pip install fastapi uvicorn websockets pyzmq

COPY main.py .

CMD ["python", "main.py"]
