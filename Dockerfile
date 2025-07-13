FROM quay.io/astronomer/astro-runtime:13.1.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .



