FROM quay.io/astronomer/astro-runtime:13.1.0


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]