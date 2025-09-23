#!/bin/bash
#SBATCH --partition=dev-gn
#SBATCH --account=sbravi
#SBATCH --nodelist=gnode02
#SBATCH --time=00:30:00
#SBATCH --job-name="download_pathways"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="download_%j.out"
#SBATCH --error="download_%j.err"
#SBATCH --mail-user=bravi.1916681@studenti.uniroma1.it
#SBATCH --mail-type=END,FAIL

echo "Hello World!"
