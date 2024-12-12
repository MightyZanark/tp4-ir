FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y openjdk-11-jre-headless openjdk-11-jdk-headless
RUN apt-get clean
RUN pip install -r requirements.txt
RUN pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git -q

RUN adduser --system --no-create-home app

EXPOSE 8080

USER app
CMD ["python3", "main.py"]
