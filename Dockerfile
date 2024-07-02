FROM python:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py

CMD ["python","app.py"]
