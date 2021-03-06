#!/bin/bash
set -e

if [ "$1" = 'run' ]; then
    python /api/src/worker.py &
    uwsgi --ini /etc/uwsgi/apps-available/api.ini
else
    exec "$@"
fi