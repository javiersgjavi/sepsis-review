#!/bin/sh

cp requirements.txt docker/requirements.txt
cd docker
docker-compose up -d

container_id=$(docker ps -aqf "name=^docker-sepsis")
docker exec -it "$container_id" bash

