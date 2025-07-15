# FROM quay.io/astronomer/astro-runtime:13.1.0
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts/ ./artifacts/

COPY . .

ENV HOST 0.0.0.0
ENV STREAMLIT_SERVER_PORT 8080

CMD ["streamlit", "run", "streamlit_app.py"]