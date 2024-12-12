FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get -y upgrade
RUN pip install -r requirements.txt
RUN adduser --system --no-create-home app

EXPOSE 8080

USER app
CMD ["python3", "main.py"]
