FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y openjdk-17-jre-headless openjdk-17-jdk-headless
RUN apt-get clean
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python3", "main.py"]
