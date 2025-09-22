#!/bin/bash -l
#SBATCH -p debug
#SBATCH --time 00:10:00
#SBATCH --job-name "download_pathways"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="download_%j.out"
#SBATCH --error="download_%j.err"

cd /home/sbravi/SBML-Codes-and-notes

docker run --rm \
  -v /home/sbravi/SBML-Codes-and-notes:/app \
  -w /app \
  sbravi/tesi \
  python download_complete_list.py