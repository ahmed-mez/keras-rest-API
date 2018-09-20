FROM python:2.7.15

# update packages
RUN apt-get update -y

# set workdir
WORKDIR /api

# pip install
RUN pip install uwsgi
COPY requirements.txt /api
RUN pip install -r requirements.txt
RUN rm requirements.txt

# deep learning weights
COPY trained_model /api/trained_model

# uwsgi conf
COPY uwsgi.ini /etc/uwsgi/apps-available/api.ini

# code 
COPY src /api/src


# entrypoint
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["run"]