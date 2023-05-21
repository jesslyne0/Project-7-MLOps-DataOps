FROM python:3.10-slim

copy requirements.txt /requirements.txt
copy . /app
WORKDIR /
ENV FLASK_APP app/main.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "flask", "run", "-host=0.0.0.0", "--reload"]