FROM quay.io/astronomer/astro-runtime:13.1.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV HOST 0.0.0.0
ENV STREAMLIT_SERVER_PORT ${PORT}

CMD ["streamlit", "run", "streamlit_app.py"]