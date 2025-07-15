# FROM quay.io/astronomer/astro-runtime:13.1.0
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV HOST 0.0.0.0
ENV STREAMLIT_SERVER_PORT 8080

CMD ["streamlit", "run", "streamlit_app.py"]