FROM quay.io/astronomer/astro-runtime:13.1.0

WORKDIR /usr/local/astro

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV HOST 0.0.0.0

CMD ["streamlit", "run", "streamlit_app.py"]