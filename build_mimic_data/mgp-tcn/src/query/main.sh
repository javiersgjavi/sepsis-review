#!/bin/sh

#Main Script to call all psql queries. 

#################################################################
#DEFINE DATABASE USERNAME: 
database=mimic
username=postgres
host=$HOST_POSTGRES
port=$POSTGRES_PORT
password=$POSTGRES_PASSWORD
#Define paths for Python script
prj_dir=../.. 

# Check if prj_dir exists, else exit with error 
if [ ! -d "$prj_dir" ]; then
    echo 'Error: the entered directory was not found' > logfile.log 
    exit 1   
fi

if [ ! -d "$prj_dir/code" ]; then
    echo 'create code directory' >> logfile.log    
    mkdir $prj_dir/code
fi
if [ ! -d "$prj_dir/code/query" ]; then
    echo 'create query directory' >> logfile.log
    mkdir $prj_dir/code/query
fi

script_dir=$prj_dir/code/query

if [ ! -d "$prj_dir/output" ]; then
    echo 'create output directory' >> logfile.log    
    mkdir $prj_dir/output
fi

out_dir=$prj_dir/output

echo 'Starting QUERY ...' 

#Sepsis-3 Query:

# Run main-query.sql (first part of main.sql up to python)


cmd="dbname=${database} user=${username} host=${host} port=${port} password=${password} options=--search_path=mimiciii"

echo 'command 1'
psql "$cmd" -f $prj_dir/src/query/main_query.sql

echo 'command 2'
# Run the Python script as this step is easier in python:
python3 $prj_dir/src/query/compute_sepsis_onset_from_exported_sql_table.py --file_timepoint $out_dir/sofa_table.csv --file_ref_time $out_dir/si_starttime.csv --sofa_threshold 2 --file_output $out_dir/sofa_delta.csv

echo 'command 3'
#Run main-write.sql (second part of main.sql after python)
psql "$cmd" -f $prj_dir/src/query/main_write.sql


echo 'command 4'
#Intermediate dynamic Python processing step to get case-control matching:
python3 match-controls.py --casefile $out_dir/q13_cases_hourly_ex1c.csv --controlfile $out_dir/q13_controls_hourly.csv --outfile $out_dir/q13_matched_controls.csv

echo 'command 5'
# Given case controls are assigned, extract relevant time windows
# Run second part of main-write (now main-write2.sql)
psql "$cmd" -f $prj_dir/src/query/main_write2.sql

echo 'finished'




