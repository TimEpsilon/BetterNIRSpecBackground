#!/bin/bash
#SBATCH -J apply_pipeline_array    # run's name
#SBATCH -N 1                       # request 1 node
#SBATCH -c 4                       
#SBATCH --mem=24GB
#SBATCH -t 10:00:00
#SBATCH -o Out.txt                 # output file name
#SBATCH -e Err.txt                 # error file name
#SBATCH --mail-type=BEGIN,END,FAIL # send me a mail at beginning and end of the job
#SBATCH --mail-user=tim.dewachter@lam.fr

working_dir="./mastDownload/JWST/" # 8 folders

file_number=$SLURM_ARRAY_TASK_ID
file_list=("$working_dir"/*) # Get a list of files in working_dir

# Check if the file_number is within the range of available files
if [ "$file_number" -ge 0 ] && [ "$file_number" -lt "${#file_list[@]}" ]; then
    file="${file_list[$file_number]}" # Get the file corresponding to file_number
    file_name=$(basename "$file") # Extract the file name
    
    # Execute the Python script and redirect output to a log file
    python -u apply_pipeline.py "$file_name" > "${file_name}_log.txt" 2>&1
else
    echo "Invalid file number: $file_number"
fi