#!/bin/sh

cp docker-compose-pg.yaml mimic-code/mimic-iii/buildmimic/docker/docker-compose.yaml
cd mimic-code/mimic-iii/buildmimic/docker # go to docker directory

dir="mimic/csv"
count=$(ls ${dir} | wc -l)

if ! [ -d "$dir" ] || ! [ $count -eq 30  ]; then
  echo "[INFO] ${dir} doesn't exist. Downloading MIMIC-III data"
  wget -r -N -c -np --user <User_of_physionet> --ask-password https://physionet.org/files/mimiciii/1.4/ # download sepsis files
  
  mkdir mimic
  mkdir mimic/csv
  mkdir mimic/pgdata
  mv physionet.org/files/mimiciii/1.4/* mimic/csv
  
else
  echo "[INFO] MIMIC-III data already exists, avoiding downloading process"
fi

echo '[INFO] Building Docker image'
sudo docker build -t postgres/mimic . # build docker image



echo '[INFO] Running Docker container'
docker-compose up

docker logs -f $(docker ps -aqf "name=^docker")
