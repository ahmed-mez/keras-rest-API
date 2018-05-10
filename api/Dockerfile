FROM python:2.7.15
RUN apt-get update -y
COPY . /api
WORKDIR /api
RUN pip install uwsgi
COPY uwsgi.ini /etc/uwsgi.ini
RUN pip install -r requirements.txt
CMD sh start.sh
