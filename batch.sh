#!/bin/bash

# Usage: ./batch.sh <path to training set> 
#                   <total number of images for subsampling>
#                   <K for K-means>
#                   <batch size>
#                   <number of epochs>
#                   <learning rate>
#                   <number of layers>
#                   <number of hidden units>

data_path=$1
num_imgs=$2
K=$3
batch_size=$4
num_epochs=$5
learning_rate=$6
num_layers=$7
num_hidden_units=$8

# Dummy vars
min_dist=0
max_dist=100000

# get current directory
curr_dir=`pwd`
proj_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# do the clustering
# suppose python=python3
python "$proj_dir/cluster.py" "$data_path" "$num_imgs" "$min_dist" "$max_dist" "$K" "$proj_dir/patient_to_labels.json" "$proj_dir/dataset/test/test.txt"

# do the training
python "$proj_dir/main.py --dataset test --batch_size $batch_size --epochs $num_epochs --lr $learning_rate --num_layers $num_layers --hidden_dim $num_hidden_units --filename results"

# aftermath
mv "performance.csv" "$num_imgs_$K_$batch_size_$num_epochs_$learning_rate_$num_layers_$num_hidden_units.csv"
mv "results" "$num_imgs_$K_$batch_size_$num_epochs_$learning_rate_$num_layers_$num_hidden_units"
