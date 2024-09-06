#!/bin/bash

MILVUS_LOG_PATH=/var/lib/milvus

kill -9 $(pgrep milvus)

docker compose down
sudo rm -rf "$MILVUS_LOG_PATH/*"
sleep 1

docker compose up -d
sleep 1
sudo chown -R $USER:$USER "$MILVUS_LOG_PATH"

