#!/bin/bash
#SBATCH -J download_data             # run's name
#SBATCH -N 1                       # request 1 node 
#SBATCH -c 1                       # request 1 cpu per task
#SBATCH --mem=8GB                 
#SBATCH -t 2:00:00               
#SBATCH -o Out.txt                 # output file name
#SBATCH -e Err.txt                 # error file name
#SBATCH --mail-type=BEGIN,END,FAIL # send me a mail at beginning and end of the job
#SBATCH --mail-user=tim.dewachter@lam.fr

python3.9 fetch_data.py