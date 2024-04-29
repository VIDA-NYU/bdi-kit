#!/bin/bash -x

#SBATCH --output=simclr-50-starmie-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=simclr-50-starmie
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yl6624@nyu.edu

# python simclr/run_pretrain.py \
#   --task arpa \
#   --batch_size 64 \
#   --lr 5e-5 \
#   --lm roberta \
#   --n_epochs 50 \
#   --max_len 128 \
#   --size 10000 \
#   --projector 768 \
#   --save_model \
#   --fp16 \
#   --sample_meth head \
#   --table_order column \
#   --run_id 0 \
#   --single_column \
#   --gpt \
#   --top_k 20

python simclr/run_pretrain.py \
  --task arpa \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 50 \
  --max_len 128 \
  --size 10000 \
  --projector 768 \
  --save_model \
  --augment_op shuffle_row,sample_row \
  --fp16 \
  --sample_meth head \
  --table_order column \
  --run_id 0 \
  --single_column \
  --top_k 50

# python simclr/run_pretrain.py --task arpa --batch_size 64 --lr 5e-5 --lm roberta --n_epochs 3 --max_len 128 --size 10000 --projector 768 --save_model --fp16 --sample_meth head --table_order column --run_id 0 --single_column --gpt
# python simclr/run_pretrain.py --task arpa --batch_size 64 --lr 5e-5 --lm roberta --n_epochs 3 --max_len 128 --size 10000 --projector 768 --augment_op shuffle_row,sample_row --fp16 --sample_meth head --table_order column --run_id 0 --single_column