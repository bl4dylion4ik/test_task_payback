FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY gradient_boosting_tune.pkl gradient_boosting_tune.pkl
COPY main.py main.py

EXPOSE 1515

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1515"]
