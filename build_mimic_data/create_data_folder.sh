#!/bin/sh

cd ..

mkdir data
mkdir data/original

mv build_mimic_data/mgp-tcn/*.pkl data/original/

sudo chmod 666 data/original/*.pkl
