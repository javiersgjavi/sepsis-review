#!/bin/sh

docker-compose up -d

container_id=$(docker ps -aqf "name=^sepsis")
docker exec -it "$container_id" bash

