#!/bin/bash -x

#SBATCH --output=askem-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=askem
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yl6624@nyu.edu

python simclr/synthetic_data_gen.py