FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
COPY .env /code

ENV PYTHONPATH=/code/src


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]