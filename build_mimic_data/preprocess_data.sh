#!/bin/sh

container_id=$(docker ps -q  --filter ancestor=postgres/mimic)
if [ ! -z "$container_id" ]; then
    docker stop $container_id && docker rm $container_id
fi

cd mgp-tcn # go to docker directory

echo 'Create image'
docker build . -t sepsis:v2

echo 'launching containers of mimic-III database and mgp-tcn repository'
cd ..
docker-compose up -d
docker exec -it $(docker ps -q  --filter ancestor=sepsis:v2) bash
