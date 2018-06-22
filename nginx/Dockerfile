FROM nginx
COPY uwsgi_params /etc/nginx/conf.d/uwsgi_params
COPY nginx.conf /etc/nginx/conf.d/default.conf
CMD nginx -c /etc/nginx/conf.d/default.conf
