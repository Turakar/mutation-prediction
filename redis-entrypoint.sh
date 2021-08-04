#!/bin/bash

/opt/redis/src/redis-server \
    --bind 127.0.0.1 \
    --protected-mode yes \
    --port 0 \
    --unixsocket redis.sock \
    --timeout 0 \
    --daemonize no \
    --supervised no \
    --databases 1 \
    --save 300 1 \
    --dbfilename $1.rdb \
    --dir /rdbs/ &
sleep 1

cd /app
eval "${@:2}"

kill -term %1
sleep 1

echo "done"
