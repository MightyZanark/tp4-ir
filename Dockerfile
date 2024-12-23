FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y openjdk-17-jre-headless openjdk-17-jdk-headless
RUN apt-get clean
RUN pip install -r requirements.txt
RUN pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git -q

RUN adduser --system --no-create-home app

EXPOSE 8080

USER app
CMD ["python3", "main.py"]
