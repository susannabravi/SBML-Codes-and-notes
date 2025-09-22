#!/bin/bash
#SBATCH --partition = dev-gn
#SBATCH --account = iacroma
#SBATCH --time 00:30:00
#SBATCH --job-name "download_pathways"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="download_%j.out"
#SBATCH --error="download_%j.err"
#SBATCH --mail-user=bravi.1916681@studenti.uniroma1.it
#SBATCH --mail-type=END,FAIL

cd /home/sbravi/SBML-Codes-and-notes

docker run --rm \
  -v /home/sbravi/SBML-Codes-and-notes:/app \
  -w /app \
  sbravi/tesi \
  python download_complete_list.py