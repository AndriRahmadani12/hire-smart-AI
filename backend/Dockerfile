FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY ./app /app/

ARG DBHOST
ARG DBPORT
ARG DBUSER
ARG DBPASS
ARG DBNAME

# Set environment variables
ENV DBHOST=${DBHOST}
ENV DBPORT=${DBPORT}
ENV DBUSER=${DBUSER}
ENV DBPASS=${DBPASS}
ENV DBNAME=${DBNAME}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007", "--reload"]
